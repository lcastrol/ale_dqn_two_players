#!/bin/bash
#A shell script to execute the ALE-Boxing experiments

nohup ./main.py --no-screen-display --saved-model-dir=exp_0_1 --log-dir-name=exp_0_1 --device=cpu --handle=play --play_epsilon=0.2 --iterations=100 --nn-file=/home/luis/ALE_Boxing/ale_dqn_two_players_testing/saved_networks_1481058035/boxing_sarsa/boxing_1201.pkl --model_file=/home/luis/ALE_Boxing/ale_dqn_two_players_testing/saved_networks_1481058035/boxing_dqn/boxing-dqn-1083478686 boxing > exp_0_1.out 2> exp_0_1.err &

nohup ./main.py --no-screen-display --saved-model-dir=exp_0_2 --log-dir-name=exp_0_2 --device=cpu --handle=play --play_epsilon=0.2 --iterations=100 --nn-file=/home/luis/ALE_Boxing/ale_dqn_two_players_testing/saved_networks_1481058035/boxing_sarsa/boxing_1201.pkl --model_file=/home/luis/ALE_Boxing/ale_dqn_two_players_testing/saved_networks_1481058035/boxing_dqn/boxing-dqn-1083478686 boxing > exp_0_2.out 2> exp_0_2.err &

nohup ./main.py --no-screen-display --saved-model-dir=exp_0_3 --log-dir-name=exp_0_3 --device=cpu --handle=play --play_epsilon=0.2 --iterations=100 --nn-file=/home/luis/ALE_Boxing/ale_dqn_two_players_testing/saved_networks_1481058035/boxing_sarsa/boxing_1201.pkl --model_file=/home/luis/ALE_Boxing/ale_dqn_two_players_testing/saved_networks_1481058035/boxing_dqn/boxing-dqn-1083478686 boxing > exp_0_3.out 2> exp_0_3.err &

