for seed in $(seq 0 4); 
    do 
        OPENAI_LOG_FORMAT=csv 
        python3 arail_trpo_cartpole.py --num_timesteps 50000 --algo trpo --seed=$seed &
    done
python3 arail_trpo_cartpole.py --num_timesteps 50000 --algo trpo --seed=5
echo 'done'
