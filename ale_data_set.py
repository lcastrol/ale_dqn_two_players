
import numpy as np
import theano

floatX = theano.config.floatX

class DataSet(object):

    def __init__(self, height, width, phi_length, rng):

        self.height = height
        self.width = width
        self.phi_length = phi_length
        self.max_steps = phi_length * 2
        self.rng = rng

        self.imgs = np.zeros((self.max_steps, self.height, self.width), dtype='uint8')
        self.actions = np.zeros(self.max_steps, dtype='int32')
        self.rewards = np.zeros(self.max_steps, dtype=floatX)

        self.top = 0 #TODO this init value is still unknown 
        self.size = 0 
        self.bottom = 0

    def add_sample(self, img, action, reward):

        self.imgs[self.top] = img 
        self.actions[self.top] = action
        self.rewards[self.top] = reward

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps 
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def phi(self, img):
        indexes = np.arange(self.top - self.phi_length + 1, self.top)
        phi = np.empty((self.phi_length, self.height, self.width), dtype=floatX)
        phi[0:self.phi_length - 1] = self.imgs.take(indexes, axis=0, mode='wrap')
        phi[-1] = img
        return phi

    def get_training_tuple(self, batch_size):

        #TODO I think this has to be fully fixed

        # Allocate the response.
        state = np.zeros((batch_size,
                          self.phi_length,
                          self.height,
                          self.width),
                         dtype=floatX)
        next_state = np.zeros((batch_size,
                          self.phi_length,
                          self.height,
                          self.width),
                         dtype=floatX)

        reward = np.zeros((batch_size,1), dtype=floatX)

        action = np.zeros((batch_size,1), dtype='int32')
        next_action = np.zeros((batch_size,1), dtype='int32') #This is called Terminals

        #terminal = np.zeros((batch_size, 1), dtype='bool')

        count = 0
        while count < batch_size:

            # Randomly choose a time step from the replay memory.
            #index = self.rng.randint(self.bottom,
            #                         self.bottom + self.size - self.phi_length)

            ## Both the before and after states contain phi_length
            ## frames, overlapping except for the first and last.
            ##all_indices = np.arange(index, index + self.phi_length + 1)
            #all_indexes = np.arange(self.top - self.phi_length, self.top)
            #end_index = index + self.phi_length - 1
            #
            ## Check that the initial state corresponds entirely to a
            ## single episode, meaning none but its last frame (the
            ## second-to-last frame in imgs) may be terminal. If the last
            ## frame of the initial state is terminal, then the last
            ## frame of the transitioned state will actually be the first
            ## frame of a new episode, which the Q learner recognizes and
            ## handles correctly during training by zeroing the
            ## discounted future reward estimate.
            ##if np.any(self.terminal.take(all_indices[0:-2], mode='wrap')):
            ##    continue

            ## Add the state transition to the response.
            #state[count] = self.imgs.take(all_indices, axis=0, mode='wrap')
            #action[count] = self.actions.take(end_index, mode='wrap')
            #reward[count] = self.rewards.take(end_index, mode='wrap')
            #terminal[count] = self.terminal.take(end_index, mode='wrap')
            
            next_indexes = np.arange(self.top - self.phi_length, self.top)
            next_state[count] = self.imgs.take(next_indexes, axis=0, mode='wrap')
            next_action[count] = self.actions[next_indexes[-1]]

            cur_indexes = next_indexes - 1 #this substracts 1 to every element of the array

            state[count] = self.imgs.take(cur_indexes, axis=0, mode='wrap')
            action[count] = self.actions[cur_indexes[-1]]
            reward[count] = self.rewards[cur_indexes[-1]]

            count += 1

        #OLD CODE ----------------------------------------------------

        #next_indexes = np.arange(self.top - self.phi_length, self.top)

        #next_state = self.imgs.take(next_indexes, axis=0, mode='wrap')
        #next_action = self.actions[next_indexes[-1]]

        #cur_indexes = next_indexes - 1 #this substracts 1 to every element of the array

        #state = self.imgs.take(cur_indexes, axis=0, mode='wrap')
        #action = self.actions[cur_indexes[-1]]
        #reward = self.rewards[cur_indexes[-1]]

        return state, action, reward, next_state, next_action
