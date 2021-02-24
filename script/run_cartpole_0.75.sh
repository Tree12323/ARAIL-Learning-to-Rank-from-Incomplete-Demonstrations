for seed in $(seq 0 4); 
    do 
        OPENAI_LOG_FORMAT=csv 
        python3 arail_trpo_cartpole.py --loss_percent 0.75 --num_timesteps 50000 --algo arail --seed=$seed &
    done
python3 arail_trpo_cartpole.py --loss_percent 0.75 --num_timesteps 50000 --algo arail --seed=5
echo 'done'
