import gym
import time
import os
import dataset
import logger
import argparse

import tensorflow as tf
import utils.tf_util as U
import numpy as np

from cg import cg
from discriminator_mujoco import Discriminator
from guidance_cartpole import Guidance
from utils.mujoco_dset import Mujoco_Dset
from utils.misc_util import set_global_seeds, zipsame, boolean_flag
from utils.math_util import explained_variance
from utils.console_util import fmt_row, colorize
from contextlib import contextmanager
from mpi4py import MPI
from collections import deque
from mpi_adam import MpiAdam
from statistics import stats
from mlp_policy_trpo import MlpPolicy

from policies import build_policy

from utils.process_expert import process_expert
from utils.process_expert import CONDITION_FULL
from utils.process_expert import CONDITION_NOISY
from utils.process_expert import CONDITION_PARTIAL
from utils.process_expert import CONDITION_PARTIAL_NOISY

from utils.input_util import observation_placeholder

global flag_render
flag_render = False
alpha = 0.5

def reward_config(true_reward, discriminator_reward, guidance_reward, algo, loss_percent, timesteps_so_far, max_timesteps):
    # return order: true reward; d_reward, g_reward
    if algo == 'trpo':
        return true_reward, true_reward, 0.0
    elif algo == 'state':
        return true_reward, alpha * discriminator_reward, 0.0
    elif algo == 'arail':
        if timesteps_so_far <= (max_timesteps // 2):
            # logger.log('no Info' + str(timesteps_so_far) + ':' + str(max_timesteps))
            return true_reward, alpha * discriminator_reward, 0.0
        else:
            # logger.log('Add Info' + str(timesteps_so_far) + ':' + str(max_timesteps))
            return true_reward, discriminator_reward, (1-loss_percent) * guidance_reward
    else:
        raise NotImplementedError

def traj_segment_generator(pi, env, reward_giver, reward_guidance, horizon, stochastic, algo, loss_percent, max_timesteps):
    global flag_render

    # Initialize state variables
    t = 0
    timesteps_so_far = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    true_rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    true_rews = np.zeros(horizon, 'float32')
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred, _, _ = pi.step(ob, stochastic=stochastic)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            timesteps_so_far = yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "ep_true_rets": ep_true_rets}
            _, vpred, _, _ = pi.step(ob, stochastic=stochastic)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        d_rew = reward_giver.get_reward(ob)
        g_rew = reward_guidance.get_rewards(agent_s=ob, agent_a=ac)
        ob, true_rew, new, _ = env.step(ac[0])

        true_rew, d_rew, g_rew = reward_config(true_reward=true_rew,
                                               discriminator_reward=d_rew,
                                               guidance_reward=g_rew,
                                               algo=algo,
                                               loss_percent=loss_percent,
                                               timesteps_so_far=timesteps_so_far,
                                               max_timesteps=max_timesteps)

        if flag_render:
            env.render()
        rews[i] = d_rew + g_rew
        true_rews[i] = true_rew

        cur_ep_ret += d_rew + g_rew
        cur_ep_true_ret += true_rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def get_vf_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'vf' in v.name[len(scope):].split('/')]

def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]

def learn(env, policy_func, reward_giver, reward_guidance, expert_dataset, rank,
          pretrained, pretrained_weight, *,
          g_step, d_step, entcoeff, save_per_iter,
          ckpt_dir, log_dir, timesteps_per_batch, task_name,
          gamma, lam, algo,
          max_kl, cg_iters, cg_damping=1e-2,
          vf_stepsize=3e-4, d_stepsize=1e-4, vf_iters=3,
          max_timesteps=0, max_episodes=0, max_iters=0, loss_percent=0.0,
          callback=None):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    policy = build_policy(env, 'mlp', value_network='copy')

    ob = observation_placeholder(ob_space)
    with tf.variable_scope('pi'):
        pi = policy(observ_placeholder=ob)
    with tf.variable_scope('oldpi'):
        oldpi = policy(observ_placeholder=ob)

    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = entcoeff * meanent

    vferr = tf.reduce_mean(tf.square(pi.vf - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = get_trainable_variables('pi')
    # var_list = [v for v in all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
    # vf_var_list = [v for v in all_var_list if v.name.startswith("pi/vff")]
    var_list = get_pi_trainable_variables("pi")
    vf_var_list = get_vf_trainable_variables("pi")
    # assert len(var_list) == len(vf_var_list) + 1
    d_adam = MpiAdam(reward_giver.get_trainable_variables())
    guidance_adam = MpiAdam(reward_guidance.get_trainable_variables())

    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(get_variables('oldpi'), get_variables('pi'))])
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    d_adam.sync()
    guidance_adam.sync()
    vfadam.sync()
    if rank == 0:
        print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, reward_giver, reward_guidance, timesteps_per_batch, stochastic=True, algo=algo, loss_percent=loss_percent, max_timesteps=max_timesteps)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards
    true_rewbuffer = deque(maxlen=40)

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    g_loss_stats = stats(loss_names)
    d_loss_stats = stats(reward_giver.loss_name)
    ep_stats = stats(["True_rewards", "Rewards", "Episode_length"])
    # if provide pretrained weight
    if pretrained_weight is not None:
        U.load_state(pretrained_weight, var_list=pi.get_variables())

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        # Save model
        if rank == 0 and iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            fname = os.path.join(ckpt_dir, task_name)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            saver = tf.train.Saver()
            saver.save(tf.get_default_session(), fname)

        logger.log("********** Iteration %i ************" % iters_so_far)

        # global flag_render
        # if iters_so_far > 0 and iters_so_far % 10 ==0:
        #     flag_render = True
        # else:
        #     flag_render = False

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        # ------------------ Update G ------------------
        logger.log("Optimizing Policy...")
        for _ in range(g_step):
            if timesteps_so_far == 0:
                with timed("sampling without Info"):
                    logger.log(str(timesteps_so_far) + ':' + str(max_timesteps))
                    seg = seg_gen.__next__()
            else:
                with timed("sampling with Info"):
                    logger.log(str(timesteps_so_far) + ':' + str(max_timesteps))
                    seg = seg_gen.send(timesteps_so_far)
            print('rewards', seg['rew'])
            add_vtarg_and_adv(seg, gamma, lam)
            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"]  # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

            args = seg["ob"], seg["ac"], atarg
            fvpargs = [arr[::5] for arr in args]

            assign_old_eq_new()  # set old parameter values to new parameter values
            with timed("computegrad"):
                *lossbefore, g = compute_lossandgrad(*args)
            lossbefore = allmean(np.array(lossbefore))
            g = allmean(g)
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")
            else:
                with timed("cg"):
                    stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank == 0)
                assert np.isfinite(stepdir).all()
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / max_kl)
                # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                surrbefore = lossbefore[0]
                stepsize = 1.0
                thbefore = get_flat()
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    set_from_flat(thnew)
                    meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                    improve = surr - surrbefore
                    logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                    if not np.isfinite(meanlosses).all():
                        logger.log("Got non-finite value of losses -- bad!")
                    elif kl > max_kl * 1.5:
                        logger.log("violated KL constraint. shrinking step.")
                    elif improve < 0:
                        logger.log("surrogate didn't improve. shrinking step.")
                    else:
                        logger.log("Stepsize OK!")
                        break
                    stepsize *= .5
                else:
                    logger.log("couldn't compute a good step")
                    set_from_flat(thbefore)
                if nworkers > 1 and iters_so_far % 20 == 0:
                    paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum()))  # list of tuples
                    assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])
            with timed("vf"):
                for _ in range(vf_iters):
                    for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                                                             include_final_partial_batch=False,
                                                             batch_size=128):
                        if hasattr(pi, "ob_rms"):
                            pi.ob_rms.update(mbob)  # update running mean/std for policy
                        g = allmean(compute_vflossandgrad(mbob, mbret))
                        vfadam.update(g, vf_stepsize)

        g_losses = meanlosses
        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        logger.log(fmt_row(13, reward_giver.loss_name))
        ob_expert, ac_expert = expert_dataset.get_next_batch(batch_size=len(ob))
        batch_size = 128
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        with timed("Discriminator"):
            for (ob_batch, ac_batch) in dataset.iterbatches((ob, ac),
                                                        include_final_partial_batch=False,
                                                        batch_size=batch_size):
                ob_expert, ac_expert = expert_dataset.get_next_batch(batch_size=batch_size)
                # update running mean/std for reward_giver
                if hasattr(reward_giver, "obs_rms"): reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
                *newlosses, g = reward_giver.lossandgrad(ob_batch, ob_expert)
                d_adam.update(allmean(g), d_stepsize)
                d_losses.append(newlosses)
        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

        # ------------------ Update Guidance ------------
        logger.log("Optimizing Guidance...")

        logger.log(fmt_row(13, reward_guidance.loss_name))
        batch_size = 128
        guidance_losses = []  # list of tuples, each of which gives the loss for a minibatch
        with timed("Guidance"):
            for ob_batch, ac_batch in dataset.iterbatches((ob, ac),
                                                        include_final_partial_batch=False,
                                                        batch_size=batch_size):
                ob_expert, ac_expert = expert_dataset.get_next_batch(batch_size=batch_size)

                idx_condition = process_expert(ob_expert, ac_expert)
                pick_idx = (idx_condition >= loss_percent)
                # pick_idx = idx_condition

                ob_expert_p = ob_expert[pick_idx]
                ac_expert_p = ac_expert[pick_idx]

                ac_batch_p = []
                for each_ob in ob_expert_p:
                    tmp_ac, _, _, _ = pi.step(each_ob, stochastic=True)
                    ac_batch_p.append(tmp_ac)

                # update running mean/std for reward_giver
                if hasattr(reward_guidance, "obs_rms"): reward_guidance.obs_rms.update(ob_expert_p)
                # reward_guidance.train(expert_s=ob_batch_p, agent_a=ac_batch_p, expert_a=ac_expert_p)
                *newlosses, g = reward_guidance.lossandgrad(ob_expert_p, ac_batch_p, ac_expert_p)
                guidance_adam.update(allmean(g), d_stepsize)
                guidance_losses.append(newlosses)
        logger.log(fmt_row(13, np.mean(guidance_losses, axis=0)))

        lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews, true_rets = map(flatten_lists, zip(*listoflrpairs))
        true_rewbuffer.extend(true_rets)
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpTrueRewMean", np.mean(true_rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens) * g_step
        arail_cur_timestep_so_far = timesteps_so_far
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank == 0:
            logger.dump_tabular()


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='CartPole-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='expert_data/cartpole')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    parser.add_argument('--loss_percent', type=float, default=0.0)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=1)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'state', 'arail'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=1e6)
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=None)
    return parser.parse_args()


def get_task_name(args):
    if args.stochastic_policy:
        task_name = args.algo + "_stochastic."
    else:
        task_name = args.algo + "_deterministic."
    if args.pretrained:
        task_name += "with_pretrained."
    task_name += args.env_id.split("-")[0]
    task_name = task_name + ".g_step_" + str(args.g_step) + ".d_step_" + str(args.d_step) + \
        ".policy_entcoeff_" + str(args.policy_entcoeff) + ".adversary_entcoeff_" + str(args.adversary_entcoeff)
    # plot_results considers digits after dash
    # at the end of the directory name to be
    # seed id and groups the runs that differ
    # only by those together
    task_name += "/seed-" + str(args.seed)
    return task_name

def get_task_short_name(args):
    task_name = args.env_id.split("-")[0] + '/'
    # plot_results considers digits after dash
    # at the end of the directory name to be
    # seed id and groups the runs that differ
    # only by those together
    if args.algo == "arail":
        task_name += "ARAIL"
    elif args.algo == "state":
        task_name += "state"
    elif args.algo == "trpo":
        task_name += "TRPO"
    else:
        raise NotImplementedError

    task_name = "%s_%.2f-"%(task_name, args.loss_percent)
    task_name += str(args.seed)
    return task_name

def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)

    task_name = get_task_short_name(args)
    logger.configure(dir='log_trpo_cartpole/%s'%task_name)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return build_policy(env, 'mlp', value_network='copy')
    import logging
    import os.path as osp
    import bench
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)

    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)

    if args.task == 'train':
        from utils.mujoco_dset import Dset_gym
        expert_observations = np.genfromtxt('expert_data/cartpole/observations.csv')
        expert_actions = np.genfromtxt('expert_data/cartpole/actions.csv', dtype=np.int32)
        expert_dataset = Dset_gym(inputs=expert_observations, labels=expert_actions, randomize=True)
        # expert_dataset = (expert_observations, expert_actions)
        reward_giver = Discriminator(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff)
        reward_guidance = Guidance(env, args.policy_hidden_size, expert_dataset=expert_dataset)
        train(env,
              args.seed,
              policy_fn,
              reward_giver,
              reward_guidance,
              expert_dataset,
              args.algo,
              args.g_step,
              args.d_step,
              args.policy_entcoeff,
              args.num_timesteps,
              args.save_per_iter,
              args.checkpoint_dir,
              args.log_dir,
              args.pretrained,
              args.BC_max_iter,
              args.loss_percent,
              task_name
              )
    elif args.task == 'evaluate':
        runner(env,
               policy_fn,
               args.load_model_path,
               timesteps_per_batch=1024,
               number_trajs=10,
               stochastic_policy=args.stochastic_policy,
               save=args.save_sample
               )
    else:
        raise NotImplementedError
    env.close()


def train(env, seed, policy_fn, reward_giver, reward_guidance, expert_dataset, algo,
          g_step, d_step, policy_entcoeff, num_timesteps, save_per_iter,
          checkpoint_dir, log_dir, pretrained, BC_max_iter, loss_percent, task_name=None):

    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):
        # Pretrain with behavior cloning
        import behavior_clone
        pretrained_weight = behavior_clone.learn(env, policy_fn, expert_dataset,
                                                 max_iters=BC_max_iter)

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env.seed(workerseed)
    learn(env, policy_fn, reward_giver, reward_guidance, expert_dataset, rank,
                    pretrained=pretrained, pretrained_weight=pretrained_weight,
                    g_step=g_step, d_step=d_step,
                    entcoeff=policy_entcoeff,
                    max_timesteps=num_timesteps,
                    ckpt_dir=checkpoint_dir, log_dir=log_dir,
                    save_per_iter=save_per_iter,
                    timesteps_per_batch=1024,
                    max_kl=0.01, cg_iters=10, cg_damping=0.1,
                    gamma=0.995, lam=0.97, algo=algo,
                    vf_iters=5, vf_stepsize=1e-3, loss_percent=loss_percent,
                    task_name=task_name)

def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    U.load_state(load_model_path)

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    from tqdm import tqdm
    for _ in tqdm(range(number_trajs)):
        traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    return avg_len, avg_ret


# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, stochastic):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []

    while True:
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj

if __name__ == "__main__":
    args = argsparser()
    main(args)
