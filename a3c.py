import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym
import multiprocessing
import os
import threading
from random import choice
from time import sleep
from time import time
import scipy.signal


def process_frame(observation):
    gray_scaled = tf.image.rgb_to_grayscale(observation, name='gray_scale')
    resized = tf.image.resize_images(
        gray_scaled, [110, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    cropped = tf.image.crop_to_bounding_box(resized, 25, 0, 84, 84)
    return cropped

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.


def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# Used to initialize weights for policy and value output layers


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


max_episode_length = 300
gamma = .99  # discount rate for advantage estimation and reward discounting
s_size = 7056  # Observations are greyscale frames of 84 * 84 * 1
a_size = 3  # Agent can move Left, Right, or Fire
load_model = False
ckpt_path = './model'
game = 'Qbert-v0'


class Network():
    """Network Class with 2 Convolutional Layers, followed by 1 Dense Layer, followed by 1 LSTM.
    Input: Frames from Atari Games
    Output: Policy and Value
    Naming from http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    slightly adapted from https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb"""

    def __init__(self, scope, action_space_size, trainer):
        with tf.variable_scope('preprocessing'):
            self.observation = tf.placeholder(
                shape=[210, 160, 3], dtype=tf.uint8)

            # rgb to grayscale
            self.gray_scaled = tf.image.rgb_to_grayscale(self.observation)

            # reshape the input
            self.resized = tf.image.resize_images(self.gray_scaled, size=[110, 84],
                                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            # this cropping was originally done in the dqn paper because there gpu implementations needed square inputs
            self.cropped = tf.image.crop_to_bounding_box(
                self.resized, 26, 0, 84, 84)

        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(
                shape=[None, 84, 84, 4], dtype=tf.float32)
            # print('hello1')
            self.input_layer = self.inputs / 255.0
            # print('hello2')
            #self.input_layer =  tf.reshape(self.inputs_scaled, [-1, 84, 84, 1])
            # print('hello3')
            self.layer1 = tf.layers.conv2d(inputs=self.input_layer,
                                           filters=16,
                                           strides=(4, 4),
                                           kernel_size=[8, 8],
                                           activation=tf.nn.relu,
                                           trainable=True)
            self.layer2 = tf.layers.conv2d(inputs=self.layer1,
                                           filters=32,
                                           strides=(2, 2),
                                           kernel_size=[4, 4],
                                           activation=tf.nn.relu,
                                           trainable=True)
            layer2_flat = tf.reshape(self.layer2, [-1, 9 * 9 * 32])

            self.layer3 = tf.layers.dense(inputs=layer2_flat,
                                          units=256,
                                          activation=tf.nn.relu,
                                          trainable=True)

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            cell_state_init = np.zeros([1, lstm_cell.state_size.c], np.float32)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros([1, lstm_cell.state_size.h], np.float32)
            self.state_init = [c_init, h_init]
            cell_state_in = tf.placeholder(
                tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (cell_state_in, h_in)
            rnn_in = tf.expand_dims(self.layer3, [0])
            step_size = tf.shape(self.input_layer)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(cell_state_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell,
                rnn_in,
                initial_state=state_in,
                sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            self.policy = tf.layers.dense(inputs=rnn_out,
                                          units=action_space_size,
                                          activation=tf.nn.softmax)
            self.value_function = tf.layers.dense(inputs=rnn_out,
                                                  units=1)

        if scope != 'global':
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(
                self.actions, action_space_size, dtype=tf.float32)
            self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

            self.responsible_outputs = tf.reduce_sum(
                self.policy * self.actions_onehot, [1])

            # Loss functions
            self.value_loss = 0.5 * \
                tf.reduce_sum(tf.square(self.target_v -
                                        tf.reshape(self.value_function, [-1])))
            self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
            self.policy_loss = - \
                tf.reduce_sum(tf.log(self.responsible_outputs)
                              * self.advantages)
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

            # Get gradients from local network using local losses
            local_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(
                self.gradients, 40.0)

            # Apply local gradients to global network
            global_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


class Worker():
    def __init__(self, game, ID, action_space_size, ckpt_path, global_episodes, trainer, gamma):
        self.name = 'worker_' + str(ID)
        self.id = ID
        self.env = gym.make(game)
        self.gamma = gamma
        self.network = Network(self.name, self.env.action_space.n, trainer)
        self.env = gym.make(game)
        self.rewards = []
        self.eps_length = []
        self.eps_mean_val = []
        self.ckpt_path = ckpt_path
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.id))
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.trainer = trainer
        self.state = []
        self.update_local_ops = update_target_graph('global', self.name)
        self.frame_buffer = []
        self.transition_buffer = []

    def train(self, episode_transitions, session, gamma, bootstrap_value):
        episode_transitions = np.array(episode_transitions)
        observations = episode_transitions[:, 0]
        actions = episode_transitions[:, 1]
        rewards = episode_transitions[:, 2]
        next_observations = episode_transitions[:, 3]
        values = episode_transitions[:, 4]

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * \
            self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)
        num_steps = len(self.frame_buffer)
        # print(num_steps)

        feed_dict_train = {
            self.network.target_v: discounted_rewards,
            self.network.inputs: np.reshape(self.frame_buffer, [num_steps, 84, 84, 4]),
            self.network.actions: actions,
            self.network.advantages: advantages,
            self.network.state_in[0]: self.batch_rnn_state[0],
            self.network.state_in[1]: self.batch_rnn_state[1]
        }

        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run(
            [self.network.value_loss,
             self.network.policy_loss,
             self.network.entropy,
             self.network.grad_norms,
             self.network.var_norms,
             self.network.state_out,
             self.network.apply_grads],
            feed_dict=feed_dict_train)

        return v_l / len(episode_transitions), p_l / len(episode_transitions), e_l / len(episode_transitions), g_n, v_n

    def work(self, max_episode_length, gamma, session, coord, saver):
        episode_count = sess.run(self.global_episodes)
        with session.as_default(), session.graph.as_default():
            while not coord.should_stop():
                done = False
                e_reward = 0
                e_steps_counter = 0
                total_steps_counter = 0
                value_buffer = []

                if e_steps_counter == 0:
                    observation = self.env.reset()
                    state_t = sess.run([self.network.cropped],
                                       {self.network.observation: observation})
                    state_t = np.reshape(state_t, [84, 84])
                    self.state = []
                    self.state += 4 * [state_t]
                    self.frame_buffer.append(self.state[:])

                state_network = self.network.state_init
                self.batch_rnn_state = state_network

                while done == False:
                    """Try changing dict layout"""
                    feed_dict_pred = {
                        self.network.state_in[0]: state_network[0],
                        self.network.state_in[1]: state_network[1],
                        self.network.inputs: np.reshape(
                            self.state, [1, 84, 84, 4])
                    }

                    action_probs, value, state_network = session.run([self.network.policy, self.network.value_function, self.network.state_out],
                                                                     feed_dict=feed_dict_pred)
                    state_temp = self.state[:]

                    action = np.random.choice(
                        action_probs[0], p=action_probs[0])
                    action = np.argmax(action_probs == action)

                    obs, reward, done, info = self.env.step(action)

                    if reward > 0:
                        reward = 1
                    elif reward < 0:
                        reward = -1

                    #next_state = process(obs)
                    next_state = sess.run([self.network.cropped],
                                          {self.network.observation: obs})

                    next_state = np.reshape(next_state, [84, 84])

                    self.state.pop(0)
                    self.state.append(next_state[:])

                    self.transition_buffer.append(
                        [state_temp, action, reward, done, value[0, 0]])
                    #print(len(self.transition_buffer), len(self.frame_buffer))
                    value_buffer.append(value[0, 0])

                    observation = next_state
                    e_reward += reward
                    e_steps_counter += 1
                    total_steps_counter += 1

                    if len(self.transition_buffer) == 4 and not done and e_steps_counter != (max_episode_length - 1):
                        #print('hello loop')

                        feed_dict_value = {
                            self.network.inputs: np.reshape(self.state, [1, 84, 84, 4]),
                            self.network.state_in[0]: state_network[0],
                            self.network.state_in[1]: state_network[1]
                        }
                        v1 = session.run(
                            self.network.value_function, feed_dict=feed_dict_value)[0, 0]
                        v_l, p_l, e_l, g_n, v_n = self.train(
                            self.transition_buffer, session, gamma, v1)
                        self.frame_buffer = []
                        self.transition_buffer = []
                        session.run(self.update_local_ops)
                    if done == True:
                        # print('What')
                        break

                    self.frame_buffer.append(self.state[:])

                self.rewards.append(e_reward)
                self.eps_length.append(e_steps_counter)
                self.eps_mean_val.append(np.mean(value_buffer))

                if len(self.transition_buffer) == 4:
                    # print('What')
                    self.train(self.transition_buffer, session, gamma, 0.0)

                if episode_count % 5 == 0 and episode_count != 0:
                    saver.save(sess, self.ckpt_path + '/model-' +
                               str(episode_count) + '.cptk')
                    print("Saved Model")

                    mean_reward = np.mean(self.rewards[-5:])
                    mean_length = np.mean(self.eps_length[-5:])
                    mean_value = np.mean(self.eps_mean_val[-5:])

                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward',
                                      simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length',
                                      simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value',
                                      simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss',
                                      simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss',
                                      simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy',
                                      simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm',
                                      simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm',
                                      simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    session.run(self.increment)
                episode_count += 1


tf.reset_default_graph()

writer = tf.summary.FileWriter(logdir="logdir")

if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

# Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
    os.makedirs('./frames')

env = gym.make(game)
action_space_size = env.action_space.n
env.close()

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(
        0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.RMSPropOptimizer(0.0001, 0.99, 0.0, 1e-6)
    # Generate global network
    master_network = Network('global', action_space_size, None)
    # Set workers to number of available CPU threads
    num_workers = multiprocessing.cpu_count()
    #num_workers = 1
    workers = []
    # Create worker classes
    for ID in range(num_workers):
        workers.append(Worker(game, ID, action_space_size,
                              ckpt_path, global_episodes, trainer, gamma))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    writer.add_graph(graph=sess.graph)

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        def worker_work(): return worker.work(
            max_episode_length, gamma, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
