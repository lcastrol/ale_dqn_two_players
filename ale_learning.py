#!/usr/bin/python
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
import pygame
import q_network

from ale_interface import AleInterface
from collections import deque
from ale_util import logger
from ale_net import DLNetwork
from sarsa_agent import SARSALambdaAgent

#pygame.init()

class DQNLearning(object):
    def __init__(self, game_name, args):

        #save game name
        self.game_name = game_name

        #Initialize logger
        self.logger = logger

        #initiallize ALE
        self.game = AleInterface(game_name, args)

        self.actions = self.game.get_actions_num()
        self.actionsB = self.game.get_actions_numB()

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

        # Screen buffer for player B
        self.buffer_length = 2
        self.buffer_count = 0
        self.screen_buffer = np.empty((self.buffer_length, args.screen_height, args.screen_width),
                                      dtype=np.uint8)

        self.frame_seq_num = args.frame_seq_num
        if args.saved_model_dir == "":
            self.param_file = "./saved_networks/%s.json" % game_name
        else:
            self.param_file = "%s/%s.json" % (args.saved_model_dir, game_name)

        # Player A
        # DQN network
        self.net = DLNetwork(game_name, self.actions, args)

        # Player B
        # SARSA 
        self.sarsa_agent = self.sarsa_init(args)

        # screen parameters
        # self.screen = (args.screen_width, args.screen_height)
        # pygame.display.set_caption(game_name)
        # self.display = pygame.display.set_mode(self.screen)

        self.deque = deque()

    #SARSA agent init 
    def sarsa_init(self, args):
        
        rng = np.random.RandomState(123456) #TODO add a random number generator

        if args.nn_file is None:
            #New network creation
            self.logger.info("Creating network for SARSA")
            #TODO a lot of missing arguments where found, now i have serious doubts about this code working
            sarsa_network = q_network.DeepQLearner(args.screen_width,
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
                                             args.epsilon_start,
                                             args.epsilon_min,
                                             args.epsilon_decay,
                                             args.experiment_prefix,
                                             rng)

        return sarsa_agent_inst

    def sarsa_get_observation(self,args):
        """ Resize and merge the previous two screen images """
        assert self.buffer_count >= 2
        index = self.buffer_count % self.buffer_length - 1
        max_image = np.maximum(self.screen_buffer[index, ...],
                               self.screen_buffer[index - 1, ...])
        return self.sarsa_resize_image(max_image, args)

    def sarsa_resize_image(self, image, args):
        """ Appropriately resize a single image """
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

    def show_screen(self, np_array):
        return
        # np_array = cv2.resize(np_array, self.screen)
        # surface = pygame.surfarray.make_surface(np_array)
        # surface = pygame.transform.rotate(surface, 270)
        # rect = pygame.draw.rect(self.display, (255, 255, 255), (0, 0, self.screen[0], self.screen[1]))
        # self.display.blit(surface, rect)
        # pygame.display.update()

    def train_net(self, args):

        self.logger.info("Training starting...")

        # training
        max_reward = 0
        epsilon, global_step = self.param_unserierlize()
        step = 0
        epoch = 0

        while True:  # loop epochs

            epoch += 1
            # initial state
            self.game.ale.reset_game()

            # two players mode switch
            self.game.ale.setMode(1) 

            # initial state sequences
            state_seq = np.empty((80, 80, self.frame_seq_num))
            for i in range(self.frame_seq_num):
                state = self.game.ale.getScreenRGB()
                self.show_screen(state)
                state = self.process_snapshot(state)
                state_seq[:, :, i] = state
            stage_reward = 0

            #TODO find a better way to do this 
            #Initiallize B screen buffer
            self.logger.info("Initiallize screen buffer for player B")
            #legal_actionsB = self.game.ale.getLegalActionSetB()

            null_reward = self.game.ale.actAB(0, 18)
            index = self.buffer_count % self.buffer_length
            self.game.ale.getScreenGrayscale(self.screen_buffer[index, ...])
            self.buffer_count += 1

            null_reward = self.game.ale.actAB(0, 18)
            index = self.buffer_count % self.buffer_length
            self.game.ale.getScreenGrayscale(self.screen_buffer[index, ...])
            self.buffer_count += 1

            #B starts the episode
            self.logger.info("Initiallize playerB")
            actionB = self.sarsa_agent.start_episode(self.sarsa_get_observation(args))

            while True:  # loop game frames

                # select action player A
                self.logger.info("Selecting player A action")

                #TODO fix predict, this was working before
                #best_act = self.net.predict([state_seq])[0]
                best_act = np.empty([2, 2])
                if random.random() <= epsilon or len(np.unique(best_act)) == 1:  # random select
                    actionA = random.randint(0, self.actions - 1)
                else:
                    actionA = np.argmax(best_act)
                #actionA = 1
                self.logger.info("Action selected for player A actionA=%d" % (actionA))

                # select action action for player B
                actionB = self.sarsa_agent.step(stage_reward, self.sarsa_get_observation(args))
                actionB += 18 #TODO fix this we must use just one value

                self.logger.info("Action selected for player B actionB=%d" % (actionB))

                # carry out selected action
                reward_n = self.game.ale.actAB(actionA, actionB)

                # get observation for player A
                state_n = self.game.ale.getScreenRGB()
                self.show_screen(state)
                state_n = self.process_snapshot(state_n)
                state_n = np.reshape(state_n, (80, 80, 1))
                state_seq_n = np.append(state_n, state_seq[:, :, : (self.frame_seq_num - 1)], axis=2)
                self.logger.info("Player A observation over")

                # get observation for player B

                #check game over state
                terminal_n = self.game.ale.game_over()

                #TODO add frame limit

                # scale down epsilon
                if step > self.observe and epsilon > self.final_epsilon:
                    epsilon -= (self.init_epsilon - self.final_epsilon) / self.explore

                # store experience
                act_onehot = np.zeros(self.actions)
                act_onehot[actionA] = 1
                self.deque.append((state_seq, act_onehot, reward_n, state_seq_n, terminal_n))
                if len(self.deque) > self.replay_memory:
                    self.deque.popleft()

                # minibatch train
                if step > self.observe and step % self.update_frequency == 0:
                    for _ in xrange(self.action_repeat):
                        mini_batch = random.sample(self.deque, self.batch_size)
                        batch_state_seq = [item[0] for item in mini_batch]
                        batch_action = [item[1] for item in mini_batch]
                        batch_reward = [item[2] for item in mini_batch]
                        batch_state_seq_n = [item[3] for item in mini_batch]
                        batch_terminal = [item[4] for item in mini_batch]
                        # predict
                        target_rewards = []
                        batch_pred_act_n = self.net.predict(batch_state_seq_n)
                        for i in xrange(len(mini_batch)):
                            if batch_terminal[i]:
                                t_r = batch_reward[i]
                            else:
                                t_r = batch_reward[i] + self.gamma * np.max(batch_pred_act_n[i])
                            target_rewards.append(t_r)
                        # train Q network
                        self.net.fit(batch_state_seq, batch_action, target_rewards)

                # update state
                state_seq = state_seq_n
                step += 1
                # serierlize param

                # save network model
                if step % self.save_model_freq == 0:
                    global_step += step
                    self.param_serierlize(epsilon, global_step)
                    self.net.save_model("%s-dqn" % self.game_name, global_step=global_step)
                    self.logger.info("save network model, global_step=%d, cur_step=%d" % (global_step, step))

                # state description
                if step < self.observe:
                    state_desc = "observe"
                elif epsilon > self.final_epsilon:
                    state_desc = "explore"
                else:
                    state_desc = "train"

                # record reward
                print "game running, step=%d, action A=%s, reward=%d, max_Q=%.6f, min_Q=%.6f" % \
                          (step, actionA, reward_n, np.max(best_act), np.min(best_act))
                if reward_n > stage_reward:
                    stage_reward = reward_n
                #END 
                if terminal_n:
                    break

            # record reward
            if stage_reward > max_reward:
                max_reward = stage_reward
            
            # log end of session
            self.logger.info(
                "epoch=%d, state=%s, step=%d(%d), max_reward=%d, epsilon=%.5f, reward=%d, max_Q=%.6f" %
                (epoch, state_desc, step, global_step, max_reward, epsilon, stage_reward, np.max(best_act)))

    def play_game(self, epsilon):

        # play games
        max_reward = 0
        epoch = 0
        if epsilon == 0.0:
            epsilon, _ = self.param_unserierlize()
        while True:  # epoch

            epoch += 1
            self.logger.info("game start...")
            # init state
            self.game.reset_game()
            # two players mode switch
            self.game.set_mode(1) 
            state_seq = np.empty((80, 80, self.frame_seq_num))
            for i in range(self.frame_seq_num):
                state = self.game.get_screen_rgb()
                self.show_screen(state)
                state = self.process_snapshot(state)
                state_seq[:, :, i] = state
            stage_step = 0
            stage_reward = 0

            while True:

                # select action
                best_act = self.net.predict([state_seq])[0]
                if random.random() < epsilon or len(np.unique(best_act)) == 1:
                    action = random.randint(0, self.actions - 1)
                else:
                    action = np.argmax(best_act)

                # get action for player B
                actionB = 19 #TODO

                # carry out selected action
                reward_n = self.game.actAB(action, actionB)
                state_n = self.game.get_screen_rgb()
                self.show_screen(state_n)
                state_n = self.process_snapshot(state_n)
                state_n = np.reshape(state_n, (80, 80, 1))
                state_seq_n = np.append(state_n, state_seq[:, :, : (self.frame_seq_num - 1)], axis=2)
                terminal_n = self.game.game_over()

                state_seq = state_seq_n
                # record
                if reward_n > stage_reward:
                    stage_reward = reward_n
                if terminal_n:
                    time.sleep(2)
                    break
                else:
                    stage_step += 1
                    stage_reward = reward_n
                    print "game running, step=%d, action=%d, reward=%d" % \
                          (stage_step, action, reward_n)

            # record reward
            if stage_reward > max_reward:
                max_reward = stage_reward

            self.logger.info("game over, epoch=%d, step=%d, reward=%d, max_reward=%d" %
                             (epoch, stage_step, stage_reward, max_reward))




#TODO remove the parsing and the executable code at the end, this will be just a library

#def parser_argument():
#    parse = argparse.ArgumentParser()
#    parse.add_argument("--play", action="store", help="play games, you can specify model file in model directory")
#    parse.add_argument("--train", action="store", help="train DQNetwork, game names is needed")
#    parse.add_argument("--gpu", action="store", default=0, help="specify gpu number")
#    args = parse.parse_args()
#    gpu = int(args.gpu)
#    if args.play is not None:
#        if not args.play.isdigit():
#            game_name = args.play
#        else:
#            game_name = "breakout"
#        dqn = DQNLearning(game_name, gpu=None)
#        dqn.play_game()
#    elif args.train is not None:
#        if not args.train.isdigit():
#            game_name = args.train
#        else:
#            game_name = "breakout"
#        dqn = DQNLearning(game_name, gpu=gpu)
#        dqn.train_net()
#    else:
#        parse.print_help()
#
#
#if __name__ == "__main__":
#    parser_argument()
