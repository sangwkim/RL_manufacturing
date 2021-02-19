import numpy as np
import tensorflow as tf
import os

"""
Critic Parameters
"""
TAU = 0.01
# LEARNING_RATE= 0.000001
GAMMA = 0.99
HIDDEN_UNITS = 512  # no. of hidden units in lstm cell
N_LAYERS = 5
TARGET_VALUE_DIMENSION = 1


class CriticNet:
    """ Critic Q value Neural Network model of the RDPG algorithm """

    def __init__(self, N_STATES, N_ACTIONS, MAX_STEP, BATCH_SIZE, lr_c):
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.MAX_STEP = MAX_STEP
        self.BATCH_SIZE = BATCH_SIZE
        self.lr_c = lr_c
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()

            """
            Critic Q network:
            """
            self.c_input_state = tf.placeholder("float", [self.MAX_STEP, None, self.N_STATES + 1],
                                                name='inputstate_placeholder')
            self.c_input_action = tf.placeholder("float", [self.MAX_STEP - 1, None, self.N_ACTIONS],
                                                 name='inputaction_placeholder')
            self.c_input_last_action = tf.placeholder("float", [1, None, self.N_ACTIONS],
                                                      name='inputlastaction_placeholder')
            self.c_input_action_merge = tf.concat([self.c_input_action, self.c_input_last_action], 0)
            self.c_input = tf.concat([self.c_input_state, self.c_input_action_merge], 2)
            self.c_target = tf.placeholder("float", [self.MAX_STEP, None, TARGET_VALUE_DIMENSION],
                                           name='label_placeholder')
            self.W_c = tf.Variable(tf.random_normal([HIDDEN_UNITS, 1]))
            self.B_c = tf.Variable(tf.random_normal([1], -0.003, 0.003))
            # lstms
            with tf.variable_scope('critic'):
                self.lstm_cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=N_LAYERS, num_units=HIDDEN_UNITS,
                                                                direction='unidirectional', dtype=tf.float32)
                self.lstm_cell.build(self.c_input.shape)
                self.lstm_outputs, self.final_state = self.lstm_cell(self.c_input)
                # self.single_lstm_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(HIDDEN_UNITS) #basiclstmcell modified to get access to cell weights
                # self.lstm_cell = CustomBasicLSTMCell(HIDDEN_UNITS) #basiclstmcell modified to get access to cell weights
                # self.lstm_layers = [self.lstm_cell]*N_LAYERS
                # self.lstm_cell = tf.contrib.rnn.MultiRNNCell([self.single_lstm_cell for _ in range(N_LAYERS)])
                # self.lstm_cell = tf.nn.rnn_cell.MultiRNNCell(self.lstm_layers,state_is_tuple = True)
                # self.init_state = self.lstm_cell.zero_state(self.BATCH_SIZE,tf.float32)
                # self.init_state = tuple(tf.contrib.rnn.LSTMStateTuple(h=tf.zeros([self.BATCH_SIZE, HIDDEN_UNITS], dtype=tf.float32),c=tf.zeros([self.BATCH_SIZE, HIDDEN_UNITS], dtype=tf.float32)) for _ in range(N_LAYERS))
                # self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(self.lstm_cell, self.c_input,initial_state = self.init_state, dtype=tf.float32,sequence_length=self.length(self.c_input))
            # self.lstm_outputs_list = tf.transpose(self.lstm_outputs, [1, 0, 2])
            self.lstm_outputs_list = tf.reshape(self.lstm_outputs, [-1, HIDDEN_UNITS])
            self.lstm_outputs_list = tf.split(self.lstm_outputs_list, self.MAX_STEP, 0)
            # prediction(output) at each time step:(list of tensors)
            self.pred_t = [tf.matmul(self.lstm_output, self.W_c) + self.B_c for self.lstm_output in
                           self.lstm_outputs_list]
            # converting target value to list of tensors
            # self.c_target_list = tf.transpose(self.c_target, [1, 0, 2])
            # self.c_target_list = tf.reshape(self.c_target_list, [-1, TARGET_VALUE_DIMENSION])
            # self.c_target_list = tf.split(self.c_target_list, self.MAX_STEP-1, 0)
            # optimizer:

            self.predt_T = tf.stack(self.pred_t)
            # transposing it to shape BatchsizeXtimestepXdimension:
            # self.predt_T = tf.transpose(self.predt_T,[1,0,2])

            self.square_diff = tf.pow((self.predt_T[-1:, :, :] - self.c_target[-1:, :, :]), 2)
            # no. of time_steps:
            # self.eff_time_step = tf.reduce_sum(self.length(self.c_input))
            # self.eff_time_step = tf.to_float(self.eff_time_step)
            # mean loss over time step:
            self.loss_t = tf.reduce_sum(self.square_diff, reduction_indices=0)
            self.loss_n = tf.reduce_sum(self.loss_t, reduction_indices=1) / self.BATCH_SIZE

            # self.last_lstm_output = self.last_relevant(self.lstm_outputs,self.length(self.c_input))
            # self.Q_last_state = tf.matmul(self.last_lstm_output, self.W_c) + self.B_c
            # print(self.lstm_cell)
            # self.params = tf.trainable_variables()
            self.params = [self.lstm_cell.weights[0], self.W_c, self.B_c]
            # self.params = [self.lstm_cell.weights[0],self.lstm_cell.weights[1],self.lstm_cell.weights[2],self.lstm_cell.weights[3],self.W_c,self.B_c]
            # self.gradient = tf.gradients(tf.pack(self.pred_t),self.params,tf.sub(self.pred_t,self.c_target_list)/(self.MAX_STEP*self.BATCH_SIZE))
            self.gradient = tf.gradients(self.loss_n, self.params)
            # self.gradient = tf.gradients(self.predt_T,self.params,(self.predt_T-self.c_target)/(self.MAX_STEP*self.BATCH_SIZE))
            self.critic_gradient = tf.gradients(self.predt_T[-1:, :, :], self.c_input_last_action)
            self.opt = tf.train.AdamOptimizer(self.lr_c)
            self.optimizer = self.opt.apply_gradients(zip(self.gradient, self.params))
            print("Initialized Critic Network...")
            """
            Target critic Q network:
            """
            # critic_q_model_parameters:
            self.t_c_input_state = tf.placeholder("float", [self.MAX_STEP, None, self.N_STATES + 1],
                                                  name='inputstate_placeholder')
            self.t_c_input_action = tf.placeholder("float", [self.MAX_STEP, None, self.N_ACTIONS],
                                                   name='inputaction_placeholder')
            self.t_c_input = tf.concat([self.t_c_input_state, self.t_c_input_action], 2)
            self.t_c_target = tf.placeholder("float", [self.MAX_STEP, None, TARGET_VALUE_DIMENSION],
                                             name='label_placeholder')
            self.t_W_c = tf.Variable(tf.random_normal([HIDDEN_UNITS, 1]))
            self.t_B_c = tf.Variable(tf.random_normal([1], -0.003, 0.003))
            # lstms
            with tf.variable_scope('target_critic'):
                self.t_lstm_cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=N_LAYERS, num_units=HIDDEN_UNITS,
                                                                  direction='unidirectional', dtype=tf.float32)
                self.t_lstm_cell.build(self.t_c_input.shape)
                self.t_lstm_outputs, self.t_final_state = self.t_lstm_cell(self.t_c_input)
                # self.t_lstm_cell = CustomBasicLSTMCell(HIDDEN_UNITS) #basiclstmcell modified to get access to cell weights
                # self.t_lstm_layers = [self.t_lstm_cell]*N_LAYERS
                # self.t_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(self.t_lstm_layers,state_is_tuple = True)
                # self.t_init_state = self.t_lstm_cell.zero_state(self.BATCH_SIZE,tf.float32)
                # self.t_lstm_outputs, self.t_final_state = tf.nn.dynamic_rnn(self.t_lstm_cell, self.t_c_input,initial_state = self.t_init_state, dtype=tf.float32,sequence_length=self.length(self.t_c_input))
            # self.t_lstm_outputs_list = tf.transpose(self.t_lstm_outputs, [1, 0, 2])
            self.t_lstm_outputs_list = tf.reshape(self.t_lstm_outputs, [-1, HIDDEN_UNITS])
            self.t_lstm_outputs_list = tf.split(self.t_lstm_outputs_list, self.MAX_STEP, 0)
            # prediction(output) at each time step:(list of tensors)
            self.t_pred_t = [tf.matmul(self.t_lstm_output, self.t_W_c) + self.t_B_c for self.t_lstm_output in
                             self.t_lstm_outputs_list]
            self.t_pred_t = tf.stack(self.t_pred_t)
            # self.t_pred_t = tf.transpose(self.t_pred_t, [1, 0, 2])
            # converting target value to list of tensors
            # self.t_c_target_list = tf.transpose(self.t_c_target, [1, 0, 2])
            # self.t_c_target_list = tf.reshape(self.t_c_target_list, [-1, TARGET_VALUE_DIMENSION])
            # self.t_c_target_list = tf.split(self.t_c_target_list, self.MAX_STEP-1, 0)
            print("Initialized Target Critic Network...")
            self.sess.run(tf.global_variables_initializer())

            # To initialize critic and target with the same values:
            # copy target parameters
            self.sess.run([
                # self.t_lstm_layers[0].weight.assign(self.lstm_layers[0].weight),
                # self.t_lstm_layers[0].bias.assign(self.lstm_layers[0].bias),
                # self.t_lstm_layers[1].weight.assign(self.lstm_layers[1].weight),
                # self.t_lstm_layers[1].bias.assign(self.lstm_layers[1].bias),
                # self.t_lstm_cell.weights[0].assign(self.lstm_cell.weights[0]),
                # self.t_lstm_cell.weights[1].assign(self.lstm_cell.weights[1]),
                # self.t_lstm_cell.weights[2].assign(self.lstm_cell.weights[2]),
                # self.t_lstm_cell.weights[3].assign(self.lstm_cell.weights[3]),
                self.t_lstm_cell.weights[0].assign(self.lstm_cell.weights[0]),
                self.t_W_c.assign(self.W_c),
                self.t_B_c.assign(self.B_c)
            ])
            # self.update_target_critic_op_ = self.t_lstm_cell.set_weights([TAU*self.lstm_cell.get_weights()[0]+(1-TAU)*self.t_lstm_cell.get_weights()[0]])
            self.update_target_critic_op = [
                # self.t_lstm_layers[0].weight.assign(TAU*self.lstm_layers[0].weight+(1-TAU)*self.t_lstm_layers[0].weight),
                # self.t_lstm_layers[0].bias.assign(TAU*self.lstm_layers[0].bias+(1-TAU)*self.t_lstm_layers[0].bias),
                # self.t_lstm_layers[1].weight.assign(TAU*self.lstm_layers[1].weight+(1-TAU)*self.t_lstm_layers[1].weight),
                # self.t_lstm_layers[1].bias.assign(TAU*self.lstm_layers[1].bias+(1-TAU)*self.t_lstm_layers[1].bias),
                # self.t_lstm_cell.weights[0].assign(TAU*self.lstm_cell.weights[0]+(1-TAU)*self.t_lstm_cell.weights[0]),
                # self.t_lstm_cell.weights[1].assign(TAU*self.lstm_cell.weights[1]+(1-TAU)*self.t_lstm_cell.weights[1]),
                # self.t_lstm_cell.weights[2].assign(TAU*self.lstm_cell.weights[2]+(1-TAU)*self.t_lstm_cell.weights[2]),
                # self.t_lstm_cell.weights[3].assign(TAU*self.lstm_cell.weights[3]+(1-TAU)*self.t_lstm_cell.weights[3]),
                self.t_lstm_cell.weights[0].assign(
                    TAU * self.lstm_cell.weights[0] + (1 - TAU) * self.t_lstm_cell.weights[0]),
                self.t_W_c.assign(TAU * self.W_c + (1 - TAU) * self.t_W_c),
                self.t_B_c.assign(TAU * self.B_c + (1 - TAU) * self.t_B_c)
            ]
            self.cwd = os.getcwd()
            self.path = os.path.join(self.cwd, 'simple') + "\\model.ckpt"
            self.saver = tf.train.Saver(max_to_keep=200)

    def length(self, data):
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def last_relevant(self, output,
                      length):  # method used while evaluating target net: where input is one or few time steps
        self.BATCH_SIZE = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        out_size = int(output.get_shape()[2])
        index = tf.range(0, self.BATCH_SIZE) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

    def train_critic(self, o_n_t, a_n_t, y_n_t):
        # print(self.sess.run([self.predt_T[:,-1:,:],self.c_target[:,-1:,:]], feed_dict={self.c_input_state: o_n_t,self.c_input_action:a_n_t[:,:-1,:],self.c_input_last_action:a_n_t[:,-1:,:],self.c_target:y_n_t}))
        self.sess.run(self.optimizer, feed_dict={self.c_input_state: o_n_t, self.c_input_action: a_n_t[:-1, :, :],
                                                 self.c_input_last_action: a_n_t[-1:, :, :], self.c_target: y_n_t})

    def compute_critic_gradient(self, o_n_t, a_n_t):  # critic gradient with respect to action
        # check = np.array(self.sess.run(self.critic_gradient, feed_dict={self.c_input_state: o_n_t,self.c_input_action: a_n_t})[0])
        # print check.shape
        # raw_input('check shape')
        return self.sess.run(self.critic_gradient,
                             feed_dict={self.c_input_state: o_n_t, self.c_input_action: a_n_t[:-1, :, :],
                                        self.c_input_last_action: a_n_t[-1:, :, :]})[0]

    def evaluate_critic(self, o_n_t, a_n_t):
        # return self.sess.run(self.pred_t, feed_dict={self.c_input_state: o_n_t,self.c_input_action:a_n_t[:,:-1,:],self.c_input_last_action:a_n_t[:,-1:,:]})[-1]
        return self.sess.run(self.pred_t, feed_dict={self.c_input_state: o_n_t, self.c_input_action: a_n_t[:-1, :, :],
                                                     self.c_input_last_action: a_n_t[-1:, :, :]})

    def evaluate_target_critic(self, o_n_t, a_n_t):
        return self.sess.run(self.t_pred_t, feed_dict={self.t_c_input_state: o_n_t, self.t_c_input_action: a_n_t})

    def update_target_critic(self):
        self.sess.run(self.update_target_critic_op)

    def save_session_critic(self, date_time, step):
        self.path = os.path.join(self.cwd, date_time) + "_critic_step%d.ckpt" % step
        self.save_path = self.saver.save(self.sess, self.path)

    def restore_session_critic(self, date_time, step):
        self.path = os.path.join(self.cwd, date_time) + "_critic_step%d.ckpt" % step
        self.saver.restore(self.sess, self.path)