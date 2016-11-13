#  -*- coding: utf-8 -*-
# based on the code of:  <yao62995@gmail.com>
# modifications by: Luis Castro and Jens Rowekamp

import os
import random
import argparse
import time
import json
import numpy as np
import cv2
import q_network

from ale_interface import AleInterface
from collections import deque
from ale_util import Logger
from ale_net import DLNetwork
from sarsa_agent import SARSALambdaAgent


class ALEtestbench(object):

    def __init__(self, args):

        # Save game name
        self.game_name = args.game

        # Initialize logger
        self._exptime = str(int(time.time()))
        self._log_dir = "./log_" + self._exptime
        self.logger = Logger(self._log_dir, self.game_name)

        # Initiallize ALE
        self.game = AleInterface(self.game_name, args)

        self.actions = self.game.get_actions_num()
        self.actionsB = self.game.get_actions_numB()

        # Set the number of iterations
        self.iterations = args.iterations

        # DQN parameters
        self.observe = args.observe
        self.explore = args.explore
        self.replay_memory = args.replay_memory
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.init_epsilon = args.init_epsilon
        self.final_epsilon = args.final_epsilon
        self.save_model_freq = args.save_model_freq
        self.update_frequency = args.update_frequency
        self.action_repeat = args.action_repeat
        self.frame_seq_num = args.frame_seq_num
        self.save_model_at_termination = args.save_model_at_termination

        # Screen buffer for player B
        self.buffer_length = 2
        self.buffer_count = 0
        self.screen_buffer = np.empty((self.buffer_length, 80, 80),
                                      dtype=np.uint8)

        # Create folder for saved NN
        args.saved_model_dir += "_" + self._exptime
        if not os.path.isdir("./" + args.saved_model_dir):
            os.makedirs("./" + args.saved_model_dir)

        # Parameters file of DQN
        self.param_file = "%s/%s.json" % (args.saved_model_dir, self.game_name)

        # Player A
        # DQN network
        self.net = DLNetwork(self.actions, self.logger, args)

        # Player B
        # SARSA learner and network
        self.sarsa_agent = self.sarsa_init(args)

        # Experience double ended queue
        self.deque = deque()

    #SARSA agent init
    def sarsa_init(self, args):

        random_seed = random.randint(0,20)
        rng = np.random.RandomState(random_seed)

        if args.nn_file is None:
            # New network creation
            self.logger.info("Creating network for SARSA")
            sarsa_network = q_network.DeepQLearner(
                                                   args.screen_width,
                                                   args.screen_height,
                                                   self.actionsB,
                                                   args.phi_length, #num_frames
                                                   args.discount,
                                                   args.learning_rate,
                                                   args.rms_decay, #rho
                                                   args.rms_epsilon,
                                                   args.momentum_sarsa,
                                                   1,#clip_delta
                                                   10000,#freeze_interval
                                                   args.batch_size,#batch_size
                                                   args.network_type,
                                                   args.update_rule,
                                                   # args.lambda_decay, #batch_accumulator
                                                   'sum',
                                                   rng)
        else:
            #Pretrained network loading
            network_file_handle = open(args.nn_file, 'r')
            sarsa_network = cPickle.load(network_file_handle)

        self.logger.info("Creating SARSA agent")
        sarsa_agent_inst = SARSALambdaAgent( sarsa_network,
                                             args,
                                             args.epsilon_min,
                                             args.epsilon_decay,
                                             args.experiment_prefix,
                                             self.logger,
                                             rng)

        return sarsa_agent_inst

    """ Merge the previous two screen images """
    def sarsa_get_observation(self,args):

        assert self.buffer_count >= 2
        index = self.buffer_count % self.buffer_length - 1
        max_image = np.maximum(self.screen_buffer[index, ...],
                               self.screen_buffer[index - 1, ...])
        return max_image
        #return self.sarsa_resize_image(max_image, args)

    """ Appropriately resize a single image """
    def sarsa_resize_image(self, image, args):

        if args.resize_method == 'crop':
            # resize keeping aspect ratio
            resize_height = int(round(
                float(self.height) * self.resized_width / self.width))

            resized = cv2.resize(image,
                                 (self.resized_width, resize_height),
                                 interpolation=cv2.INTER_LINEAR)

            # Crop the part we want
            crop_y_cutoff = resize_height - CROP_OFFSET - self.resized_height
            cropped = resized[crop_y_cutoff:
                              crop_y_cutoff + self.resized_height, :]

            return cropped
        elif args.resize_method == 'scale':
            return cv2.resize(image,
                              (args.screen_width, args.screen_height),
                              interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError('Unrecognized image resize method.')

    def param_serierlize(self, epsilon, step):
        json.dump({"epsilon": epsilon, "step": step}, open(self.param_file, "w"))

    def param_unserierlize(self):
        if os.path.exists(self.param_file):
            jd = json.load(open(self.param_file, 'r'))
            return jd['epsilon'], jd["step"]
        else:
            return self.init_epsilon, 0

    def process_snapshot(self, snap_shot):
        # rgb to gray, and resize
        snap_shot = cv2.cvtColor(cv2.resize(snap_shot, (80, 80)), cv2.COLOR_BGR2GRAY)
        # image binary
        # _, snap_shot = cv2.threshold(snap_shot, 1, 255, cv2.THRESH_BINARY)
        return snap_shot

    """ Training function, this is the main loop for training """
    def train_net(self, args):

        self.logger.info("Running training experiment")
        self.logger.info("Player A (DQN agent), settings:")
        self.logger.info("Initial Epsilon: %.6f" % (args.init_epsilon))
        self.logger.info("Epsilon decay rate: 1/%.6f epsilon per episode" % (args.explore))
        self.logger.info("Final Epsilon: %.6f" % (args.final_epsilon))
        self.logger.info("Training starting...")

        self.logger.info("Player A (SARSA agent), settings:")

        # Initiallize variables
        max_reward = 0
        epsilon, global_step = self.param_unserierlize()
        step = 0
        epoch = 0
        max_game_iterations = self.iterations

        # Epochs loop
        while True:

            # Initiallize variables
            #step = 0
            game_dif_score = 0
            playerA_score = 0
            playerB_score = 0

            # initial state
            self.game.ale.reset_game()

            # Two players mode switch ON
            self.game.ale.setMode(1)

            # Initial state sequences
            state_seq = np.empty((args.screen_width, args.screen_height, self.frame_seq_num))
            for i in range(self.frame_seq_num):
                state = self.game.ale.getScreenRGB()
                state = self.process_snapshot(state)
                state_seq[:, :, i] = state
            stage_reward = 0

            # Player B initiallization flag
            playerB_is_uninitiallized = True

            #Episode loop, each turn on the loop is a "step" and it implies a new game frame
            while True:

                # Select action player A
                self.logger.info("Selecting player A action")
                best_act = self.net.predict([state_seq])[0]

                # Prevent player A to take actions on the first two frames to add fairness
                if step < 2:
                    actionA = 0
                else:
                    # Normal epsilon-greedy policy
                    if (random.random() <= epsilon) or (len(np.unique(best_act)) == 1):
                        actionA = random.randint(0, self.actions - 1) # random action selection
                    else:
                        actionA = np.argmax(best_act) # Best action selection

                self.logger.info("Action selected for player A actionA=%d" % (actionA))

                # Select action player B
                self.logger.info("Selecting player B action")
                if self.buffer_count >= self.buffer_length+1:
                    if (playerB_is_uninitiallized == True):
                        self.logger.info("Initiallize playerB")
                        actionB = self.sarsa_agent.start_episode(self.sarsa_get_observation(args))
                        actionB += 18 #TODO again fix this, it is anoying!!
                        playerB_is_uninitiallized = False
                    else:
                        actionB = self.sarsa_agent.step(-reward_n, playerB_observation)
                        actionB += 18 #TODO again fix this, it is anoying!!
                else:
                    actionB = 18 #TODO fix this we must use just one value

                self.logger.info("Action selected for player B actionB=%d" % (actionB))

                # Carry out selected actions
                self.logger.info("Executing actions")
                reward_n = self.game.ale.actAB(actionA, actionB)

                #Save_scores
                game_dif_score += reward_n
                if (reward_n > 0):
                    playerA_score += reward_n
                elif (reward_n < 0):
                    playerB_score += reward_n

                # Getting screen image
                state_n = self.game.ale.getScreenRGB()
                state_n_grayscale = self.process_snapshot(state_n)

                # Get observation for player A
                state_n = np.reshape(state_n_grayscale, (80, 80, 1))
                state_seq_n = np.append(state_n, state_seq[:, :, : (self.frame_seq_num - 1)], axis=2)
                self.logger.info("Player A observation over")

                # Get observation for player B
                screen_buffer_index = self.buffer_count % self.buffer_length
                self.screen_buffer[screen_buffer_index, ...] = state_n_grayscale
                # Wait until the buffer is full
                if self.buffer_count >= self.buffer_length:
                    playerB_observation = self.sarsa_get_observation(args)
                # Reset buffer counter to avoid overflow
                if self.buffer_count == (10*self.buffer_length):
                    self.buffer_count = self.buffer_length + 1
                else:
                    self.buffer_count += 1
                self.logger.info("Player B observation over")

                # Check game over state
                terminal_n = self.game.ale.game_over()

                #TODO add frame limit

                # Scale down epsilon
                if (step > self.observe) and (epsilon > self.final_epsilon):
                    epsilon -= (self.init_epsilon - self.final_epsilon) / self.explore

                # Store experience
                act_onehot = np.zeros(self.actions)
                act_onehot[actionA] = 1
                self.deque.append((state_seq, act_onehot, reward_n, state_seq_n, terminal_n))
                if len(self.deque) > self.replay_memory:
                    self.deque.popleft()

                # DQN Minibatch train
                if ((step > self.observe) and ((step % self.update_frequency) == 0)):
                    for _ in xrange(self.action_repeat):
                        mini_batch = random.sample(self.deque, self.batch_size)
                        batch_state_seq = [item[0] for item in mini_batch]
                        batch_action = [item[1] for item in mini_batch]
                        batch_reward = [item[2] for item in mini_batch]
                        batch_state_seq_n = [item[3] for item in mini_batch]
                        batch_terminal = [item[4] for item in mini_batch]

                        # Predict
                        target_rewards = []
                        batch_pred_act_n = self.net.predict(batch_state_seq_n)

                        for i in xrange(len(mini_batch)):
                            if batch_terminal[i]:
                                t_r = batch_reward[i]
                            else:
                                t_r = batch_reward[i] + self.gamma * np.max(batch_pred_act_n[i])
                            target_rewards.append(t_r)
                        # Train Q network
                        self.net.fit(batch_state_seq, batch_action, target_rewards)

                # Update state sequence
                state_seq = state_seq_n

                # Increase step counter
                step += 1

                # Save network model if using the save_model_freq param
                if ((step % self.save_model_freq) == 0) and (self.save_model_at_termination == False):
                    global_step += step
                    self.param_serierlize(epsilon, global_step)
                    self.net.save_model("%s-dqn" % self.game_name, global_step=global_step)
                    self.logger.info("Saving network model, global_step=%d, cur_step=%d" % (global_step, step))

                # Player A state description
                if step < self.observe:
                    state_desc = "observe"
                elif epsilon > self.final_epsilon:
                    state_desc = "explore"
                else:
                    state_desc = "train"

                # Record step information
                self.logger.exp([epoch, step, actionA, actionB, reward_n, game_dif_score, playerA_score, playerB_score, epsilon])

                # Record max episode reward
                if reward_n > stage_reward:
                    stage_reward = reward_n

                # End the episode
                if terminal_n:
                    self.logger.info("Episode %d is over" % epoch)
                    # Save network models
                    self.sarsa_agent.end_episode(-reward_n)
                    if(self.save_model_at_termination == True):
                        global_step += step
                        self.param_serierlize(epsilon, global_step)
                        self.net.save_model("%s-dqn" % self.game_name, global_step=global_step)
                        self.logger.info("Saving network model, global_step=%d, cur_step=%d" % (global_step, step))
                    break
                #****************** END episode loop ****************************************

            # Record max reward
            if stage_reward > max_reward:
                max_reward = stage_reward

            # Record epoch information
            #self.logger.exp([epoch, max_reward, epsilon])

            # Log end of epoch
            self.logger.info(
                "epoch=%d, state=%s, step=%d(%d), max_reward=%d, epsilon=%.5f, reward=%d, max_Q=%.6f" %
                (epoch, state_desc, step, global_step, max_reward, epsilon, stage_reward, np.max(best_act)))

            # Increase epoch counter
            epoch += 1

            # Break the loop after max_game_iterations
            if epoch >= max_game_iterations:
                self.sarsa_agent.finish_epoch(epoch)
                break
            #****************** END epoch loop ****************************************

    """ Play function, this is the main loop for using pretrained networks """
    def play_game(self, epsilon):

        # Init vars
        max_reward = 0
        epoch = 0

        #Load epsilon value
        if epsilon == 0.0:
            epsilon, _ = self.param_unserierlize()

        # Epochs loop
        while True:

            self.logger.info("Episode start...")

            # Reset the game in ale
            self.game.ale.reset_game()

            # two players mode switch
            self.game.ale.setMode(1)

            # Init state sequence
            state_seq = np.empty((args.screen_width, args.screen_height, self.frame_seq_num))
            for i in range(self.frame_seq_num):
                state = self.game.get_screen_rgb()
                state = self.process_snapshot(state)
                state_seq[:, :, i] = state

            # Init
            stage_step = 0
            stage_reward = 0

            # Episodes loop
            while True:

                # Get action player A
                best_act = self.net.predict([state_seq])[0]
                if random.random() < epsilon or len(np.unique(best_act)) == 1:
                    action = random.randint(0, self.actions - 1)
                else:
                    action = np.argmax(best_act)

                # Get action player B
                self.logger.info("Selecting player B action")
                if self.buffer_count >= self.buffer_length+1:
                    if (playerB_is_uninitiallized == True):
                        self.logger.info("Initiallize playerB")
                        actionB = self.sarsa_agent.start_episode(self.sarsa_get_observation(args))
                        actionB += 18 #TODO again fix this, it is anoying!!
                        playerB_is_uninitiallized = False
                    else:
                        actionB = self.sarsa_agent.step(-reward_n, playerB_observation)
                        actionB += 18 #TODO again fix this, it is anoying!!
                else:
                    actionB = 18 #TODO fix this we must use just one value
                #actionB = 19 #
                self.logger.info("Action selected for player B actionB=%d" % (actionB))

                # Carry out selected action
                reward_n = self.game.actAB(action, actionB)

                # Get observation player A
                state_n = self.game.get_screen_rgb()
                state_n = self.process_snapshot(state_n)
                state_n = np.reshape(state_n, (80, 80, 1))
                state_seq_n = np.append(state_n, state_seq[:, :, : (self.frame_seq_num - 1)], axis=2)

                # Check game over state
                terminal_n = self.game.game_over()

                # Update state sequence
                state_seq = state_seq_n

                # Record episode max reward #This makes no sense since max reward is 2
                #if reward_n > stage_reward:
                #    stage_reward = reward_n

                # Increase step counter
                stage_step += 1

                # End he episode
                if terminal_n:
                    self.logger.info("Episode %d is over" % epoch)
                    time.sleep(2)
                    break
                #****************** END episode loop ****************************************

            # Record global max reward
            if stage_reward > max_reward:
                max_reward = stage_reward

            self.logger.info("game over, epoch=%d, step=%d, reward=%d, max_reward=%d" %
                             (epoch, stage_step, stage_reward, max_reward))

            #Increase the epochs counter
            epoch += 1

            # Break the loop after max_game_iterations
            if epoch >= max_game_iterations:
                self.sarsa_agent.finish_epoch(epoch)
                break
            #****************** END epoch loop ****************************************
