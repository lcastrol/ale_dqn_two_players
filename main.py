#!/usr/bin/python2
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com>
# modifications by Luis Castro and Jens Rowekamp

from defaults import defaults

import argparse
from ale_learning import ALEtestbench

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def parser_argument():
    parse = argparse.ArgumentParser()

    # ------------------------------------------------------------------------------------------
    # Mandatory arguments
    # ------------------------------------------------------------------------------------------
    parse.add_argument("game", type=str, help="game name")

    # ------------------------------------------------------------------------------------------
    # Experiment arguments
    # ------------------------------------------------------------------------------------------
    parse.add_argument("--handle", type=str, help="\"train\" or \"play\"")
    parse.add_argument("--iterations", type=int, default=defaults.EPOCHS, help="number of game iterations to play / train")

    # ------------------------------------------------------------------------------------------
    # ALE arguments
    # ------------------------------------------------------------------------------------------
    parse.add_argument("--no-screen-display", dest="display_screen", action='store_false', default=True, help="Turn off screen display")
    parse.add_argument("--frame_skip", type=int, default=4, help="frame skip number")
    parse.add_argument("--repeat_action_probability", type=float, default=0, help="repeat action probability")
    parse.add_argument("--color_averaging", type=str2bool, default=True, help="color average")
    parse.add_argument("--random_seed", type=int, default=0, help="random seed")

    # ------------------------------------------------------------------------------------------
    # DQN arguments
    # ------------------------------------------------------------------------------------------
    # DQN epsilon arguments
    parse.add_argument("--observe", type=int, default=defaults.DQN_OBSERVE_LIMIT, help="Number of steps before start the DQN training stage")
    parse.add_argument("--explore", type=float, default=defaults.DQN_EXPLORE_LIMIT, help="Number of steps before start the DQN explore stage")
    parse.add_argument("--init_epsilon", type=float, default=1.0, help="Initial value for DQN epsilon")
    parse.add_argument("--final_epsilon", type=float, default=0.1, help="Final value of DQN epsilon")

    parse.add_argument("--replay_memory", type=int, default=50000, help="")
    parse.add_argument("--gamma", type=float, default=0.99, help="")
    parse.add_argument("--update_frequency", type=int, default=defaults.UPDATE_FREQUENCY, help=" Frequency of the minibatch train for DQN in steps")
    parse.add_argument("--action_repeat", type=int, default=4, help="")

    parse.add_argument("--device", type=str, default="gpu", help="cpu or gpu")
    parse.add_argument("--gpu", type=int, default=0, help="gpu average")
    parse.add_argument("--batch_size", type=int, default=32, help="batch size")
    parse.add_argument("--optimizer", choices=['rmsprop', 'adam', 'sgd'], default='rmsprop', help='Network optimization algorithm')
    parse.add_argument("--learn_rate", type=float, default=0.00025, help="Learning rate")
    parse.add_argument("--decay_rate", type=float, default=0.95, help="decay rate, used for Rmsprop")
    parse.add_argument("--momentum", type=float, default=0.95, help="momentum, used for Rmsprop")

    parse.add_argument("--with_pool_layer", type=str2bool, default=False, help="whether has max_pool layer")
    parse.add_argument("--frame_seq_num", type=int, default=4, help="frame seq number")
    parse.add_argument("--saved_model_dir", type=str, default="saved_networks", help="")
    parse.add_argument("--model_file", type=str, default="", help="")
    parse.add_argument("--save_model_freq", type=int, default=defaults.SAVE_MODEL_FREQ, help="")
    parse.add_argument("--save-model-at-termination", dest="save_model_at_termination", action='store_true', default=False, help="False|True, save the DQN model at the termination of the episode")

    # ------------------------------------------------------------------------------------------
    # Sarsa arguments
    # ------------------------------------------------------------------------------------------
    parse.add_argument('--nn-file', dest="nn_file", type=str, default=None, help='Pickle file containing trained net.')

    parse.add_argument("--screen-width", dest="screen_width", type=int, default=defaults.RESIZED_WIDTH, help="resize screen width")
    parse.add_argument("--screen-height", dest="screen_height", type=int, default=defaults.RESIZED_HEIGHT, help="resize screen height")

    parse.add_argument('--phi-length', dest="phi_length", type=int, default=4, help=('Number of recent frames used to represent ' + 'state. (default: 4)'))
    parse.add_argument('--discount', type=float, default=.99, help='Discount rate (default: .99)')
    parse.add_argument('--learning-rate', dest="learning_rate", type=float, default=.00025, help='Learning rate (default: .00025 )')
    parse.add_argument('--rms-decay', dest="rms_decay", type=float, default=defaults.RMS_DECAY, help='Decay rate for rms_prop (default: %(default)s)')
    parse.add_argument('--rms-epsilon', dest="rms_epsilon", type=float, default=defaults.RMS_EPSILON, help='Denominator epsilson for rms_prop ' + '(default: )')
    parse.add_argument('--momentum-sarsa', dest="momentum_sarsa", type=float, default=defaults.MOMENTUM, help=('Momentum term for Nesterov momentum. '+ '(default: %(default)s)')) #TODO check the other momentum
    parse.add_argument('--network-type', dest="network_type", type=str, default=defaults.NETWORK_TYPE, help=('nips_cuda|nips_dnn|nature_cuda|nature_dnn' + '|linear (default: %(default)s)'))
    parse.add_argument('--update-rule', dest="update_rule", type=str, default=defaults.UPDATE_RULE, help=('deepmind_rmsprop|rmsprop|sgd ' + '(default: %(default)s)'))
    parse.add_argument('--lambda', dest="lambda_decay", type=float, default=defaults.LAMBDA, help=('Lambda value. ' + '(default: %(default)s)'))

    parse.add_argument('--epsilon-start', dest="epsilon_start", type=float, default=defaults.EPSILON_START, help=('Starting value for epsilon. ' + '(default: %(default)s)'))
    parse.add_argument('--epsilon-min', dest="epsilon_min", type=float, default=defaults.EPSILON_MIN, help='Minimum epsilon. (default: %(default)s)')
    parse.add_argument('--epsilon-decay', dest="epsilon_decay", type=float, default=defaults.EPSILON_DECAY, help=('Number of steps to minimum epsilon. ' + '(default: %(default)s)'))

    parse.add_argument('--experiment-prefix', dest="experiment_prefix", default=None, help='Experiment name prefix ' '(default is the name of the game)')
    parse.add_argument('--resize-method', dest="resize_method", type=str, default=defaults.RESIZE_METHOD, help=('crop|scale (default: %(default)s)'))

    # parameter for play
    parse.add_argument("--play_epsilon", type=float, default=0.0, help="a float value in [0, 1), 0 means use global train epsilon")

    #Parse the arguments
    args = parse.parse_args()

    #TODO Fix this, this should trigger a warning at least
    if args.handle != "train":  # use cpu when play games
        args.device = "cpu"

    #Assign a prefix
    #TODO add a time stamp to the prefix
    if args.experiment_prefix is None:
        args.experiment_prefix = args.game

    #Generate the experiment wrapper
    ale_testbench = ALEtestbench(args)

    #Trigger the experiment depending on the mode play|train
    if args.handle == "train":
        ale_testbench.train_net(args)
    else:
        ale_testbench.play_game(args.play_epsilon)

if __name__ == "__main__":
    parser_argument()
