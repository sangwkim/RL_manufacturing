import numpy as np
from actor_net import ActorNet
from critic_net import CriticNet
import os
import datetime

BUFFER_SIZE = 75000
GAMMA = 0.99


class RDPG:
    """Recurrent Policy Gradient Algorithm"""

    def __init__(self, N_STATES, N_ACTIONS, STEPS, BATCH_SIZE, lr_c, lr_a):
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.STEPS = STEPS
        self.BATCH_SIZE = BATCH_SIZE
        self.lr_c = lr_c
        self.lr_a = lr_a
        self.critic_net = CriticNet(self.N_STATES, self.N_ACTIONS, self.STEPS - 2, self.BATCH_SIZE, self.lr_c)
        self.actor_net = ActorNet(self.N_STATES, self.N_ACTIONS, self.STEPS - 2, self.BATCH_SIZE, self.lr_a)
        self.full_history = np.empty([0, 1 + self.N_STATES + self.N_ACTIONS])
        self.state_indice = np.zeros([self.STEPS - 2, 1, self.N_STATES])
        self.action_indice = np.zeros([self.STEPS - 2, 1, self.N_ACTIONS])
        self.state_indice_test = np.zeros([self.STEPS, 10, self.N_STATES])
        self.action_indice_test = np.zeros([self.STEPS, 10, self.N_ACTIONS])
        self.when = np.reshape(np.linspace(100, 0, self.STEPS - 2), [self.STEPS - 2, 1, 1])
        self.when_batch = np.repeat(np.reshape(np.linspace(100, 0, self.STEPS - 2), [self.STEPS - 2, 1, 1]),
                                    self.BATCH_SIZE, axis=1)
        self.when_test = np.repeat(np.reshape(np.linspace(100, 0, self.STEPS - 2), [self.STEPS - 2, 1, 1]),
                                   10, axis=1)

        self.cwd = os.getcwd()

    def evaluate_actor(self, act, obs):
        # converting state to a 3D tensor to feed into lstms
        self.action_indice[:-1, 0, :] = self.action_indice[1:, 0, :]
        self.action_indice[-1, 0, :] = act
        self.state_indice[:-1, 0, :] = self.state_indice[1:, 0, :]
        self.state_indice[-1, 0, :] = obs
        # print(self.state_matrix[1:,0,:])
        # print(self.state_matrix[1:,1,:])
        return self.actor_net.evaluate_actor(np.concatenate((self.action_indice, self.state_indice, self.when), 2))[0]

    def compute_action(self, hh):
        self.state_indice_test = hh[:, :, 1:-self.N_ACTIONS]
        self.action_indice_test = hh[:, :, -self.N_ACTIONS:]

        return self.actor_net.evaluate_actor(
            np.concatenate((self.action_indice_test[1:-1, :, :], self.state_indice_test[2:, :, :], self.when_test), 2))

    def compute_Q(self, hh):
        self.state_indice_test = hh[:, :, 1:-self.N_ACTIONS]
        self.action_indice_test = hh[:, :, -self.N_ACTIONS:]
        self.q_ht = self.critic_net.evaluate_critic(
            np.concatenate((self.state_indice_test[2:, :, :], self.when_test), 2), self.action_indice_test[2:, :, :])
        return self.q_ht[-1]

    def log_history(self, roa):

        self.full_history = np.append(self.full_history, roa, 0)

        if len(self.full_history) > BUFFER_SIZE:
            self.full_history = np.delete(self.full_history, 0, 0)

    def store_transition_offline(self, M):

        self.full_history = M

    def sample_mini_batches(self):

        self.indices = np.random.randint(0, self.full_history.shape[0], size=(self.BATCH_SIZE))
        self.R_mini_batch = [None] * self.BATCH_SIZE

        for i in range(0, len(self.indices)):

            self.R_mini_batch[i] = self.full_history[max(0, self.indices[i] - self.STEPS):max(0, self.indices[
                i] - self.STEPS) + self.STEPS, :]

            zero_indice = np.where(self.R_mini_batch[i][:, 1] == 0)[0]

            if zero_indice.size > 1:
                print("error: more than one zero indices in the sequence")
            elif zero_indice.size == 1:
                self.indices[i] = self.indices[i] + self.STEPS
                self.R_mini_batch[i] = self.full_history[self.indices[i] - self.STEPS:self.indices[i], :]

        # reward_t (batchsize x timestep)
        self.r_n_tl = [None] * self.BATCH_SIZE
        for i in range(0, len(self.r_n_tl)):
            self.r_n_tl[i] = self.R_mini_batch[i][:, 0]

        self.r_n_t = np.zeros([self.BATCH_SIZE, self.STEPS])

        for i in range(0, self.BATCH_SIZE):
            self.r_n_t[i, 0:len(self.r_n_tl[i])] = self.r_n_tl[i]

        self.r_n_t = self.r_n_t.transpose([1, 0])

        # observation list (batchsize x timestep)
        self.o_n_tl = [None] * self.BATCH_SIZE
        for i in range(0, len(self.o_n_tl)):
            self.o_n_tl[i] = self.R_mini_batch[i][:, 1:1 + self.N_STATES]

        self.o_n_t = np.zeros([self.BATCH_SIZE, self.STEPS, self.N_STATES])
        for i in range(0, self.BATCH_SIZE):
            self.o_n_t[i, 0:len(self.o_n_tl[i]), :] = self.o_n_tl[i]

        self.o_n_t = self.o_n_t.transpose([1, 0, 2])
        self.e_n_t = np.expand_dims(self.o_n_t[:, :, 1] - self.o_n_t[:, :, 3], 2)
        np.concatenate((self.o_n_t, self.e_n_t), 2)

        # action list:
        # observation list (batchsize x timestep)
        self.a_n_tl = [None] * self.BATCH_SIZE
        for i in range(0, len(self.a_n_tl)):
            self.a_n_tl[i] = self.R_mini_batch[i][:, -self.N_ACTIONS:]

        self.a_n_t = np.zeros([self.BATCH_SIZE, self.STEPS, self.N_ACTIONS])
        for i in range(0, self.BATCH_SIZE):
            self.a_n_t[i, 0:len(self.a_n_tl[i]), :] = self.a_n_tl[i]

        self.a_n_t = self.a_n_t.transpose([1, 0, 2])

    def train(self):
        self.sample_mini_batches()
        # Action at h_t+1:
        self.t_a_ht1 = self.actor_net.evaluate_target_actor(
            np.concatenate((self.a_n_t[1:-1, :, :], self.o_n_t[2:, :, :], self.when_batch), 2))
        self.t_a_ht_in = np.concatenate((self.a_n_t[2:-1, :, :], self.t_a_ht1[-1:, :, :]), 0)
        # State Action value at h_t+1:

        self.t_qht1 = self.critic_net.evaluate_target_critic(np.concatenate((self.o_n_t[2:, :, :], self.when_batch), 2),
                                                             self.t_a_ht_in)
        self.check = self.t_qht1

        ##COMPUTE TARGET VALUES FOR EACH SAMPLE EPISODE (y_1,y_2,....y_t) USING THE RECURRENT TARGET NETWORKS
        self.y_n_t = []
        self.r_n_t = np.reshape(self.r_n_t, [self.STEPS, self.BATCH_SIZE, 1])

        for i in range(0, self.STEPS - 2):
            self.y_n_t.append(self.r_n_t[i + 2, :, :] + GAMMA * self.t_qht1[i, :, :])
        self.y_n_t = np.hstack(self.y_n_t)
        self.y_n_t = self.y_n_t.transpose([1, 0])
        self.y_n_t = np.reshape(self.y_n_t, [self.STEPS - 2, self.BATCH_SIZE, 1])

        # self.y_n_t = np.vstack(self.y_n_t)
        # self.y_n_t = self.y_n_t.T #(batchsize x timestep)
        # self.y_n_t = np.reshape(self.y_n_t,[self.BATCH_SIZE,self.STEPS-1,1]) #reshape y_n_t to have shape (batchsize,timestep,no.dimensions)
        ##COMPUTE CRITIC UPDATE (USING BPTT)
        self.critic_net.train_critic(np.concatenate((self.o_n_t[1:-1, :, :], self.when_batch), 2),
                                     self.a_n_t[1:-1, :, :], self.y_n_t)
        # action for computing critic gradient
        self.a_ht = self.actor_net.evaluate_actor_batch(
            np.concatenate((self.a_n_t[:-2, :, :], self.o_n_t[1:-1, :, :], self.when_batch),
                           2))  # returns output as 3d array
        self.a_ht_in = np.concatenate((self.a_n_t[1:-2, :, :], self.a_ht[-1:, :, :]), 0)
        # self.a_ht_in = self.a_n_t[:,:-1,:]
        # self.a_ht_in[:,-1:,:] = self.a_ht[:,-1:,:]
        # critic gradient with respect to action delQ/dela
        self.del_Q_a = self.critic_net.compute_critic_gradient(
            np.concatenate((self.o_n_t[1:-1, :, :], self.when_batch), 2), self.a_ht_in)
        ##COMPUTE ACTOR UPDATE (USING BPTT)
        self.actor_net.train_actor(np.concatenate((self.a_n_t[:-2, :, :], self.o_n_t[1:-1, :, :], self.when_batch), 2),
                                   self.del_Q_a)
        ##Update the target networks
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()

    def save_session(self, date_time, step):
        np.save(os.path.join(self.cwd, date_time) + "step%d.npy" % step, np.array(self.full_history))
        self.actor_net.save_session_actor(date_time, step)
        self.critic_net.save_session_critic(date_time, step)

    def restore_session(self, date_time, step):
        self.path = os.path.join(self.cwd, date_time) + "step%d.ckpt" % step
        self.full_history = np.load(os.path.join(self.cwd, date_time) + "step%d.npy" % step)
        self.pointer = step
        self.actor_net.restore_session_actor(date_time, step)
        self.critic_net.restore_session_critic(date_time, step)