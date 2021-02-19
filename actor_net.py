import numpy as np
import tensorflow as tf
import os

"""
Actor Parameters
"""
TAU = 0.01
# LEARNING_RATE= 0.00001

GAMMA = 0.99
HIDDEN_UNITS = 512  # no. of hidden units in lstm cell
N_LAYERS = 5
TARGET_VALUE_DIMENSION = 1


class ActorNet:
    """ Actor Neural Network model of the RDPG algorithm """

    def __init__(self, N_STATES, N_ACTIONS, MAX_STEP, BATCH_SIZE, lr_a):
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.MAX_STEP = MAX_STEP
        self.BATCH_SIZE = BATCH_SIZE
        self.lr_a = lr_a
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()

            """
            Actor network:
            """
            self.a_input_states = tf.placeholder("float", [self.MAX_STEP, None, self.N_ACTIONS + self.N_STATES + 1],
                                                 name='input_placeholder')
            self.a_grad_from_critic = tf.placeholder("float", [1, None, self.N_ACTIONS], name='input_placeholder')

            self.W_a = tf.Variable(tf.random_normal([HIDDEN_UNITS, self.N_ACTIONS]))
            self.B_a = tf.Variable(tf.random_normal([1, self.N_ACTIONS], -0.003, 0.003))
            # lstms
            with tf.variable_scope('actor'):
                self.lstm_cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=N_LAYERS, num_units=HIDDEN_UNITS,
                                                                direction='unidirectional', dtype=tf.float32)
                self.lstm_cell.build(self.a_input_states.shape)
                self.lstm_outputs, self.final_state = self.lstm_cell(self.a_input_states)
                # self.lstm_cell = CustomBasicLSTMCell(HIDDEN_UNITS) #basiclstmcell modified to get access to cell weights
                # self.lstm_layers = [self.lstm_cell]*N_LAYERS
                # self.lstm_cell = tf.nn.rnn_cell.MultiRNNCell(self.lstm_layers,state_is_tuple = True)
                # self.init_state = self.lstm_cell.zero_state(self.BATCH_SIZE,tf.float32)
                # self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(self.lstm_cell, self.a_input_states,initial_state = self.init_state, dtype=tf.float32,sequence_length=self.length(self.a_input_states))
            # self.lstm_outputs_list = tf.transpose(self.lstm_outputs, [1, 0, 2])
            self.lstm_outputs_list = tf.reshape(self.lstm_outputs, [-1, HIDDEN_UNITS])
            self.lstm_outputs_list = tf.split(self.lstm_outputs_list, self.MAX_STEP, 0)
            # prediction(output) at each time step:(list of tensors)
            self.pred_t = [tf.matmul(self.lstm_output, self.W_a) + self.B_a for self.lstm_output in
                           self.lstm_outputs_list]
            self.pred_t_array = tf.stack(self.pred_t)
            # self.pred_t_array = tf.transpose(self.pred_t_array, [1, 0, 2]) #(to get shape of batch sizexstepxdimension)
            """
            last relevant action (while evaluating actor during testing)
            """
            # self.last_lstm_output = self.last_relevant(self.lstm_outputs,self.length(self.a_input_states))
            # self.action_last_state = tf.multiply(tf.sigmoid(tf.multiply(tf.matmul(self.last_lstm_output, self.W_a) + self.B_a,0.01)),100)
            # optimizer:
            # self.params = tf.trainable_variables()
            self.params = [self.lstm_cell.weights[0], self.W_a, self.B_a]
            # self.a_grad_from_criticT = tf.transpose(self.a_grad_from_critic,perm=[1,0,2])

            incr = self.a_grad_from_critic > 0
            self.inverted_dQda = tf.where(incr, tf.where(self.pred_t_array[-1:, :, :] < 100, self.a_grad_from_critic,
                                                         self.a_grad_from_critic * (
                                                                     100. - self.pred_t_array[-1:, :, :]) / 100.),
                                          tf.where(self.pred_t_array[-1:, :, :] > 0, self.a_grad_from_critic,
                                                   self.a_grad_from_critic * self.pred_t_array[-1:, :, :] / 100.))
            self.gradient = tf.gradients(self.pred_t_array[-1:, :, :], self.params, -self.inverted_dQda / BATCH_SIZE)

            # self.gradient = tf.gradients(self.pred_t_array[-1:,:,:],self.params,-self.a_grad_from_critic/BATCH_SIZE)#- because we are interested in maximization
            self.gradient_a = tf.gradients(self.pred_t_array[-1:, :, :], self.a_input_states)
            self.opt = tf.train.AdamOptimizer(self.lr_a)
            self.optimizer = self.opt.apply_gradients(zip(self.gradient, self.params))
            print("Initialized Actor Network...")

            """
            Target Actor network:
            """
            self.t_a_input_states = tf.placeholder("float", [self.MAX_STEP, None, self.N_ACTIONS + self.N_STATES + 1],
                                                   name='input_placeholder')
            self.t_a_grad_from_critic = tf.placeholder("float", [self.MAX_STEP, None, self.N_ACTIONS],
                                                       name='input_placeholder')

            self.t_W_a = tf.Variable(tf.random_normal([HIDDEN_UNITS, self.N_ACTIONS]))
            self.t_B_a = tf.Variable(tf.random_normal([1, self.N_ACTIONS], -0.003, 0.003))
            # lstms
            with tf.variable_scope('target_actor'):
                self.t_lstm_cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=N_LAYERS, num_units=HIDDEN_UNITS,
                                                                  direction='unidirectional', dtype=tf.float32)
                self.t_lstm_cell.build(self.t_a_input_states.shape)
                self.t_lstm_outputs, self.t_final_state = self.t_lstm_cell(self.t_a_input_states)
                # self.t_lstm_cell = CustomBasicLSTMCell(HIDDEN_UNITS) #basiclstmcell modified to get access to cell weights
                # self.t_lstm_layers = [self.t_lstm_cell]*N_LAYERS
                # self.t_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(self.t_lstm_layers,state_is_tuple = True)
                # self.t_init_state = self.lstm_cell.zero_state(self.BATCH_SIZE,tf.float32)
                # self.t_lstm_outputs, self.t_final_state = tf.nn.dynamic_rnn(self.t_lstm_cell, self.t_a_input_states,initial_state = self.t_init_state, dtype=tf.float32,sequence_length=self.length(self.t_a_input_states))
            # self.t_lstm_outputs_list = tf.transpose(self.t_lstm_outputs, [1, 0, 2])
            self.t_lstm_outputs_list = tf.reshape(self.t_lstm_outputs, [-1, HIDDEN_UNITS])
            self.t_lstm_outputs_list = tf.split(self.t_lstm_outputs_list, self.MAX_STEP, 0)
            # prediction(output) at each time step:(list of tensors)
            self.t_pred_t = [tf.matmul(self.t_lstm_output, self.t_W_a) + self.t_B_a for self.t_lstm_output in
                             self.t_lstm_outputs_list]
            self.t_pred_t = tf.stack(self.t_pred_t)
            # self.t_pred_t = tf.transpose(self.t_pred_t, [1, 0, 2]) #(to get shape of batch sizexstepxdimension)
            """
            last relevant action (while evaluating actor during testing)
            """
            # self.t_last_lstm_output = self.last_relevant(self.t_lstm_outputs,self.length(self.t_a_input_states))
            # self.t_action_last_state = tf.multiply(tf.sigmoid(tf.multiply(tf.matmul(self.t_last_lstm_output, self.t_W_a) + self.t_B_a,0.01)),100)
            print("Initialized Target Actor Network...")
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
                self.t_W_a.assign(self.W_a),
                self.t_B_a.assign(self.B_a)
            ])

            self.update_target_actor_op = [
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
                self.t_W_a.assign(TAU * self.W_a + (1 - TAU) * self.t_W_a),
                self.t_B_a.assign(TAU * self.B_a + (1 - TAU) * self.t_B_a)
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
        L_BATCH_SIZE = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        out_size = int(output.get_shape()[2])
        index = tf.range(0, L_BATCH_SIZE) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

    def actor_gradient(self, o_n_t):
        return self.sess.run(self.gradient_a, feed_dict={self.a_input_states: o_n_t})

    def train_actor(self, o_n_t, del_Q_a):
        self.sess.run(self.optimizer, feed_dict={self.a_input_states: o_n_t, self.a_grad_from_critic: del_Q_a})

    def evaluate_actor(self, o_n_t):
        return self.sess.run(self.pred_t, feed_dict={self.a_input_states: o_n_t})[-1]

    def evaluate_actor_batch(self, o_n_t):
        return self.sess.run(self.pred_t_array, feed_dict={self.a_input_states: o_n_t})

    def evaluate_target_actor(self, o_n_t):
        return self.sess.run(self.t_pred_t, feed_dict={self.t_a_input_states: o_n_t})

    def update_target_actor(self):
        self.sess.run(self.update_target_actor_op)

    def save_session_actor(self, date_time, step):
        self.path = os.path.join(self.cwd, date_time) + "_actor_step%d.ckpt" % step
        self.save_path = self.saver.save(self.sess, self.path)

    def restore_session_actor(self, date_time, step):
        self.path = os.path.join(self.cwd, date_time) + "_actor_step%d.ckpt" % step
        self.saver.restore(self.sess, self.path)