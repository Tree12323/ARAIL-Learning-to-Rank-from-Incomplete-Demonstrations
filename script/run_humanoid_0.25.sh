for seed in $(seq 0 4);
    do
        OPENAI_LOG_FORMAT=csv
        python3 arail_trpo.py --stochastic --num_timestep 1000000 --loss_percent 0.25 --algo arail --seed=$seed --env_id Humanoid-v2 --expert_path expert_data/mujoco/stochastic.trpo.Humanoid.0.00.npz &
    done
python3 arail_trpo.py --stochastic --num_timestep 1000000 --loss_percent 0.25 --algo arail --seed=5 --env_id Humanoid-v2 --expert_path expert_data/mujoco/stochastic.trpo.Humanoid.0.00.npz
