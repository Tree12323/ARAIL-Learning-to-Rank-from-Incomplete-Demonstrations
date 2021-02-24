for seed in $(seq 0 4);
    do
        OPENAI_LOG_FORMAT=csv
        python3 arail_trpo.py --stochastic --num_timestep 1000000 --loss_percent 0 --algo trpo --seed=$seed --env_id Hopper-v2 --expert_path expert_data/mujoco/stochastic.trpo.Hopper.0.00.npz &
    done
python3 arail_trpo.py --stochastic --num_timestep 1000000 --loss_percent 0 --algo trpo --seed=5 --env_id Hopper-v2 --expert_path expert_data/mujoco/stochastic.trpo.Hopper.0.00.npz
