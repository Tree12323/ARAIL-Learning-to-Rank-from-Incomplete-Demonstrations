import copy
import numpy as np
import tensorflow as tf
import utils.tf_util as U
from utils.misc_util import RunningMeanStd

def RankLoss(predict_score1, predict_score2, label, margin=0.5):
    return tf.nn.relu(tf.multiply(tf.subtract(predict_score1, predict_score2), label) + margin)

class Guidance:
    def __init__(self, env, hidden_size, expert_dataset):
        self.hidden_size = hidden_size
        self.expert_dataset = expert_dataset
        with tf.variable_scope('guidance'):
            self.scope = tf.get_variable_scope().name

            self.agent_s = tf.placeholder(dtype=tf.float32, 
                                          shape=[None] + list(env.observation_space.shape),
                                          name='ph_agent_s')
            self.agent_a = tf.placeholder(dtype=tf.float32, 
                                          shape=[None] + list(env.action_space.shape), 
                                          name='ph_agent_a')
            self.expert_a = tf.placeholder(dtype=tf.float32, 
                                           shape=[None] + list(env.action_space.shape),
                                           name='ph_expert_a')

            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=env.observation_space.shape)
            obs_ph_rms = (self.agent_s - self.obs_rms.mean) / self.obs_rms.std

            layer_s = tf.layers.dense(inputs=obs_ph_rms,
                                      units=self.hidden_size, 
                                      activation=tf.nn.leaky_relu, 
                                      name='layer_s')

            layer_a = tf.layers.dense(inputs=self.agent_a, 
                                      units=self.hidden_size, 
                                      activation=tf.nn.leaky_relu, 
                                      name='layer_a')

            layer_s_a = tf.concat([layer_s, layer_a], axis=1)

            layer = tf.layers.dense(inputs=layer_s_a, 
                                    units=self.hidden_size, 
                                    activation=tf.nn.leaky_relu, 
                                    name='layer1')

            output = tf.layers.dense(inputs=layer, 
                                     units=env.action_space.shape[0], 
                                     activation=tf.identity, 
                                     name='layer2')

            ##########
            # BUG
            ##########
            # loss_func = tf.contrib.gan.losses.wargs.mutual_information_penalty
            labels = tf.nn.softmax(self.expert_a)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))

            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(self.loss)

        self.loss_name = ["guidance_loss"]
        var_list = self.get_trainable_variables()
        self.lossandgrad = U.function([self.agent_s, self.agent_a, self.expert_a],
                                      [self.loss] + [U.flatgrad(self.loss, var_list)])

    def train(self, expert_s, agent_a, expert_a):
        return tf.get_default_session().run(self.train_op, feed_dict={self.expert_a: expert_a,
                                                                      self.agent_s: expert_s,
                                                                      self.agent_a: agent_a})

    def get_rewards(self, agent_s, agent_a):
        expert_a = []
        if len(agent_s.shape) == 1:
            agent_s = np.expand_dims(agent_s, 0)
        if len(agent_a.shape) == 1:
            agent_a = np.expand_dims(agent_a, 0)
        for each_s in agent_s:
            # tmp_expert_a = self.expert_dataset.find_nearest_action(each_s)
            tmp_expert_a = self.expert_dataset.sample_action(each_s)
            expert_a.append(tmp_expert_a)
        return 1./ (1e-3 + tf.get_default_session().run(self.loss, feed_dict={self.agent_s: agent_s, 
                                                                   self.agent_a: agent_a, 
                                                                   self.expert_a: expert_a}))

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class Guidance_Rank:
    def __init__(self, env, hidden_size, expert_dataset):
        with tf.variable_scope('guidance'):
            self.hidden_size = hidden_size
            self.scope = tf.get_variable_scope().name
            self.observation_shape = env.observation_space.shape
            self.actions_shape = env.action_space.shape

            self.build_ph()

            # Build grpah
            # 构建图(生成图和专家图)输出1维
            generator_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
            expert_logits = self.build_graph(self.generator_obs_ph, self.expert_acs_ph, reuse=True)

            self.expert_label = tf.ones_like(expert_logits)
            self.generator_label = tf.zeros_like(generator_logits)

            # label
            label = self.generator_label >= self.expert_label
            label = (tf.cast(label, tf.float32) - 0.5) * 2

            loss = RankLoss(
                predict_score1=generator_logits,
                predict_score2=expert_logits,
                label=label
            )

            self.loss = tf.reduce_mean(loss)

            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(self.loss)
        # Build Reward for policy 为什么用生成器作为reward_op
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        self.loss_name = ["guidance__loss"]
        var_list = self.get_trainable_variables()
        self.lossandgrad = U.function([self.generator_obs_ph, self.generator_acs_ph, self.expert_acs_ph],
                                      [self.loss] + [U.flatgrad(self.loss, var_list)])

    # 建立占位符(e_obs,e_acs,g_obs,g_acs)
    def build_ph(self):
        self.generator_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape, name="r_observations_ph")
        self.generator_acs_ph = tf.placeholder(tf.float32, (None,) + self.actions_shape, name="r_actions_ph")
        self.expert_acs_ph = tf.placeholder(tf.float32, (None,) + self.actions_shape, name="r_expert_actions_ph")

    # 构建图：三层全连接
    def build_graph(self, obs_ph, acs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # 状态标准化
            with tf.variable_scope("guide_obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)
            obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
            # 合并
            _input = tf.concat([obs, acs_ph], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return logits

    def train(self, expert_s, agent_a, expert_a):
        return tf.get_default_session().run(self.train_op, feed_dict={self.expert_acs_ph: expert_a,
                                                                      self.generator_obs_ph: expert_s,
                                                                      self.generator_acs_ph: agent_a})

    def get_rewards(self, obs, acs):
        sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)
        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: acs}
        reward = sess.run(self.reward_op, feed_dict)
        return reward

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)