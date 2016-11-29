#!/bin/bash
#A shell script to execute the ALE-Boxing experiments

./main.py --no-screen-display --saved-model-dir=exp_8_3 --log-dir-name=exp_8_3 --device=cpu --handle=play --play_epsilon=0.8 --iterations=30 --nn-file=/home/luis/ALE_Boxing/ale_dqn_two_players/saved_networks_1479689039/boxing_sarsa/boxing_200.pkl --model_file=/home/luis/ALE_Boxing/ale_dqn_two_players/saved_networks_1479689039/boxing_dqn/boxing-dqn-143534100 boxing &
./main.py --no-screen-display --saved-model-dir=exp_8_4 --log-dir-name=exp_8_5 --device=cpu --handle=play --play_epsilon=0.8 --iterations=30 --nn-file=/home/luis/ALE_Boxing/ale_dqn_two_players/saved_networks_1479689039/boxing_sarsa/boxing_200.pkl --model_file=/home/luis/ALE_Boxing/ale_dqn_two_players/saved_networks_1479689039/boxing_dqn/boxing-dqn-143534100 boxing & 
