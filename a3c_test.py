import tensorflow as tf
import numpy as np
import threading
import gym
import operator
import itertools
from matplotlib import pyplot as plt
import multiprocessing
import os
import math
from multiprocessing import Process
from time import sleep
"""Changes:
Deleted global var initializer
Deleted with sess.as_default"""
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("number_ps", 0, "Number of parameter servers")

FLAGS = tf.app.flags.FLAGS
job_name = FLAGS.job_name
num_ps = FLAGS.number_ps

nodes_address = []
ps_list = []
workers_list = []
worker_n = 0
with open('node_list.txt') as nodes:
    for node in nodes:
        node = node.strip('\n')
        nodes_address.append(node)

total_num_nodes = len(nodes_address)
for i, node in enumerate(nodes_address[:num_ps]):
    print(type(node), node)
    print(FLAGS.task_index)
    if node == str(FLAGS.task_index):
        print('Parameter Server index')
        worker_n = i
    print('Parameter Server added ' + str(i))
    ps_list.append("icsnode" + node + ".cluster.net:2222")

for i, node in enumerate(nodes_address[num_ps:]):
    print(type(node), node)
    print(FLAGS.task_index)
    if node == str(FLAGS.task_index):
        print('Worker Server index')
        worker_n = i
    print('Worker Server added ' + str(i))
    workers_list.append("icsnode" + node + ".cluster.net:2222")


cluster = tf.train.ClusterSpec({
    "worker": workers_list,
    "ps": ps_list
})
server = tf.train.Server(
    cluster,
    job_name=FLAGS.job_name,
    task_index=worker_n)


def copy_network(from_scope, to_scope):
    global_val = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    local_val = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    copy_val = []
    for g, w in zip(local_val, global_val):
        op = g.assign(w)
        copy_val.append(op)

    return copy_val

# Include learning rate


def make_train_op(local_estimator, global_estimator, clip_norm):
    local_grads, _ = zip(*local_estimator.gradients)
    # Clip gradients
    local_grads, _ = tf.clip_by_global_norm(local_grads, clip_norm)
    _, global_vars = zip(*global_estimator.gradients)
    local_global_grads_and_vars = list(zip(local_grads, global_vars))
    return global_estimator.optimizer.apply_gradients(local_global_grads_and_vars,
                                                      global_step=tf.train.get_global_step())


class PolicyValueNetwork():
    """
    Used to create the graph of the network. The network will have two heads, one for policy and other for value.

    Args:
        number of actions in policy.
    """

    def __init__(self, num_actions, scope_input):
        with tf.variable_scope("preprocessing"):
            self.observation = tf.placeholder(
                shape=[210, 160, 3], dtype=tf.uint8)

            # rgb to grayscale
            self.proc_state = tf.image.rgb_to_grayscale(self.observation)

            # reshape the input
            self.proc_state = tf.image.resize_images(self.proc_state, size=[110, 84],
                                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            # this cropping was originally done in the dqn paper because there gpu implementations needed square inputs
            self.proc_state = tf.image.crop_to_bounding_box(
                self.proc_state, 25, 0, 84, 84)

        with tf.variable_scope("hidden_layers"):
            # Placeholder for stacked processed states
            self.state_u = tf.placeholder(
                shape=[None, 84, 84, 4], dtype=tf.float32)

            self.state = tf.to_float(self.state_u) / 255.0

            # First conv layer with 16 fliters, 8x8 size of filter, 4 stride of the filter, with ReLu
            self.conv1 = tf.contrib.layers.conv2d(
                self.state, 16, 8, 4, activation_fn=tf.nn.relu, trainable=True, weights_initializer=tf.contrib.layers.xavier_initializer())

            # Second conv layer with 32 filters, 4x4 and stride of 2, with ReLu
            self.conv2 = tf.contrib.layers.conv2d(
                self.conv1, 32, 4, 2, activation_fn=tf.nn.relu, trainable=True, weights_initializer=tf.contrib.layers.xavier_initializer())

            # flatten conv output
            self.conv2_flat = tf.contrib.layers.flatten(self.conv2)

            # Fully connected layer with 256 units and ReLu
            self.fc1 = tf.contrib.layers.fully_connected(
                self.conv2_flat, 256, activation_fn=tf.nn.relu, trainable=True, weights_initializer=tf.contrib.layers.xavier_initializer())

            # summaries
            # tf.contrib.layers.summarize_activation(self.conv1)
            # tf.contrib.layers.summarize_activation(self.conv2)
            # tf.contrib.layers.summarize_activation(self.fc1)

        # Network for policy (state-action function)
        with tf.variable_scope("policy_net"):
            # fully connected layer with number of outputs = number of actions
            self.fc2 = tf.contrib.layers.fully_connected(
                self.fc1, num_actions, activation_fn=None, trainable=True, weights_initializer=tf.contrib.layers.xavier_initializer())
            # Soft max over the outputs
            self.state_action = tf.contrib.layers.softmax(self.fc2) + 1e-20
            # squeeze to remove all the 1's from the shape
            self.policy = tf.squeeze(self.state_action)

        with tf.variable_scope("value_net"):
            self.value = tf.contrib.layers.fully_connected(
                self.fc1, 1, activation_fn=None, trainable=True, weights_initializer=tf.contrib.layers.xavier_initializer())
            self.value_transpose = tf.transpose(self.value)

        with tf.variable_scope("loss_calculation"):
            self.advantage = tf.placeholder(
                shape=[None, None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None, None], dtype=tf.int32)
            self.actions = tf.squeeze(self.actions)
            self.actions_onehot = tf.squeeze(tf.one_hot(
                self.actions, num_actions, dtype=tf.float32))
            self.reward = tf.placeholder(shape=[None, None], dtype=tf.float32)

            # logging
            self.episode_reward = tf.Variable(
                0, name="episode_reward", dtype=tf.float32)

            # policy network loss
            self.entropy = - \
                tf.reduce_sum(self.state_action * tf.log(self.state_action), 1)

            # adding a small value to avoid NaN's
            self.log_pi = tf.log(self.state_action)
            self.log_prob_actions = tf.reduce_sum(
                tf.multiply(self.log_pi, self.actions_onehot), 1)
            self.policy_loss = - \
                tf.reduce_sum(self.log_prob_actions *
                              self.advantage + 0.01 * self.entropy)

            # value network loss
            self.value_loss = 0.5 * \
                tf.nn.l2_loss(self.reward - self.value_transpose)

            # total loss
            self.loss = self.value_loss + self.policy_loss

        with tf.variable_scope("optimization"):
            self.learning_rate = tf.placeholder(shape=[], dtype=tf.float32)
            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate, 0.99, 0.0, 1e-6)
            self.gradients = self.optimizer.compute_gradients(self.loss)
            self.gradients = [[grad, var]
                              for grad, var in self.gradients if grad is not None]
            self.gradients_apply = self.optimizer.apply_gradients(self.gradients,
                                                                  global_step=tf.train.get_global_step())

        # summary
        #tf.summary.scalar("Total_loss", self.loss)
        #tf.summary.scalar("Entropy", self.entropy)
        #tf.summary.scalar("Policy loss", self.policy_loss)
        #tf.summary.scalar("Value loss", self.value_loss)
        tf.summary.scalar(self.episode_reward.op.name, self.episode_reward)

        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        sumaries = [s for s in summary_ops if "global" in s.name]
        sumaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.summary.merge(sumaries)


class GlobalPolicyValueNetwork():
    """
    Used to create the graph of the network. The network will have two heads, one for policy and other for value.

    Args:
        number of actions in policy.
    """

    def __init__(self, num_actions, scope_input):
        with tf.variable_scope("preprocessing"):
            self.observation = tf.placeholder(
                shape=[210, 160, 3], dtype=tf.uint8)

            # rgb to grayscale
            self.proc_state = tf.image.rgb_to_grayscale(self.observation)

            # reshape the input
            self.proc_state = tf.image.resize_images(self.proc_state, size=[110, 84],
                                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            # this cropping was originally done in the dqn paper because there gpu implementations needed square inputs
            self.proc_state = tf.image.crop_to_bounding_box(
                self.proc_state, 25, 0, 84, 84)
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % worker_n,
                cluster=cluster)):
            with tf.variable_scope("hidden_layers"):
                # Placeholder for stacked processed states
                self.state_u = tf.placeholder(
                    shape=[None, 84, 84, 4], dtype=tf.float32)

                self.state = tf.to_float(self.state_u) / 255.0

                # First conv layer with 16 fliters, 8x8 size of filter, 4 stride of the filter, with ReLu
                self.conv1 = tf.contrib.layers.conv2d(
                    self.state, 16, 8, 4, activation_fn=tf.nn.relu, trainable=True, weights_initializer=tf.contrib.layers.xavier_initializer())

                # Second conv layer with 32 filters, 4x4 and stride of 2, with ReLu
                self.conv2 = tf.contrib.layers.conv2d(
                    self.conv1, 32, 4, 2, activation_fn=tf.nn.relu, trainable=True, weights_initializer=tf.contrib.layers.xavier_initializer())

                # flatten conv output
                self.conv2_flat = tf.contrib.layers.flatten(self.conv2)

                # Fully connected layer with 256 units and ReLu
                self.fc1 = tf.contrib.layers.fully_connected(
                    self.conv2_flat, 256, activation_fn=tf.nn.relu, trainable=True, weights_initializer=tf.contrib.layers.xavier_initializer())

                # summaries
                # tf.contrib.layers.summarize_activation(self.conv1)
                # tf.contrib.layers.summarize_activation(self.conv2)
                # tf.contrib.layers.summarize_activation(self.fc1)

            # Network for policy (state-action function)
            with tf.variable_scope("policy_net"):
                # fully connected layer with number of outputs = number of actions
                self.fc2 = tf.contrib.layers.fully_connected(
                    self.fc1, num_actions, activation_fn=None, trainable=True, weights_initializer=tf.contrib.layers.xavier_initializer())
                # Soft max over the outputs
                self.state_action = tf.contrib.layers.softmax(self.fc2) + 1e-20
                # squeeze to remove all the 1's from the shape
                self.policy = tf.squeeze(self.state_action)

            with tf.variable_scope("value_net"):
                self.value = tf.contrib.layers.fully_connected(
                    self.fc1, 1, activation_fn=None, trainable=True, weights_initializer=tf.contrib.layers.xavier_initializer())
                self.value_transpose = tf.transpose(self.value)

            with tf.variable_scope("loss_calculation"):
                self.advantage = tf.placeholder(
                    shape=[None, None], dtype=tf.float32)
                self.actions = tf.placeholder(
                    shape=[None, None], dtype=tf.int32)
                self.actions = tf.squeeze(self.actions)
                self.actions_onehot = tf.squeeze(tf.one_hot(
                    self.actions, num_actions, dtype=tf.float32))
                self.reward = tf.placeholder(
                    shape=[None, None], dtype=tf.float32)

                # logging
                self.episode_reward = tf.Variable(
                    0, name="episode_reward", dtype=tf.float32)

                # policy network loss
                self.entropy = - \
                    tf.reduce_sum(self.state_action *
                                  tf.log(self.state_action), 1)

                # adding a small value to avoid NaN's
                self.log_pi = tf.log(self.state_action)
                self.log_prob_actions = tf.reduce_sum(
                    tf.multiply(self.log_pi, self.actions_onehot), 1)
                self.policy_loss = - \
                    tf.reduce_sum(self.log_prob_actions *
                                  self.advantage + 0.01 * self.entropy)

                # value network loss
                self.value_loss = 0.5 * \
                    tf.nn.l2_loss(self.reward - self.value_transpose)

                # total loss
                self.loss = self.value_loss + self.policy_loss

            with tf.variable_scope("optimization"):
                self.learning_rate = tf.placeholder(shape=[], dtype=tf.float32)
                self.optimizer = tf.train.RMSPropOptimizer(
                    self.learning_rate, 0.99, 0.0, 1e-6)
                self.gradients = self.optimizer.compute_gradients(self.loss)
                self.gradients = [[grad, var]
                                  for grad, var in self.gradients if grad is not None]
                self.gradients_apply = self.optimizer.apply_gradients(self.gradients,
                                                                      global_step=tf.train.get_global_step())

            # summary
            #tf.summary.scalar("Total_loss", self.loss)
            #tf.summary.scalar("Entropy", self.entropy)
            #tf.summary.scalar("Policy loss", self.policy_loss)
            #tf.summary.scalar("Value loss", self.value_loss)
            tf.summary.scalar(self.episode_reward.op.name, self.episode_reward)

            var_scope_name = tf.get_variable_scope().name
            summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
            sumaries = [s for s in summary_ops if "global" in s.name]
            sumaries = [s for s in summary_ops if var_scope_name in s.name]
            self.summaries = tf.summary.merge(sumaries)


class Worker():
    def __init__(self, game, id, t_max, num_actions, global_network, gamma, summary_writer,
                 learning_rate, max_global_time_step, clip_norm, global_counter, episode_counter):

        self.action = []
        self.value_state = []
        self.state_buffer = []
        self.state = []
        self.reward = []
        self.r_return = []

        self.episode_state = []
        # current steps of the worker
        self.steps_worker = 0
        self.episode_reward = 0

        self.game = game
        self.id = id
        self.t_max = t_max
        self.num_actions = num_actions
        self.initial_learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.discount = gamma

        self.done = False
        self.writer = summary_writer

        self.max_global_time_step = max_global_time_step
        self.global_step = tf.train.get_global_step()
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.episode_counter = episode_counter

        # Initialise the environment
        self.env = gym.make(self.game)

        self.global_network = global_network
        # Worker network
        with tf.variable_scope(self.id):
            self.w_network = PolicyValueNetwork(self.num_actions, self.id)

        # Cannot change uninitialised graph concurrently.
        self.copy_network = copy_network("global", self.id)

        self.grad_apply = make_train_op(self.w_network, self.global_network,
                                        self.clip_norm)  # include learning rate as one of the input

    def play(self, coord, sess, saver, CHECKPOINT_DIR):
        learning_rate = self.initial_learning_rate
        global_t = 0
        episode_count = 0
        count = 0
        c_lives = 4
        lives = 4

        while not coord.should_stop():

            sess.run(self.copy_network)
            t = 0
            # if self.steps_worker < 100000:
            #     lives = 4
            # create a state buffer from a single state and append it to state buffer
            if self.done or self.steps_worker == 0 or (c_lives != lives):
                # if self.done or self.steps_worker == 0:
                observation = self.env.reset()
                proccessed_state = sess.run([self.w_network.proc_state],
                                            {self.w_network.observation: observation})
                proccessed_state = np.reshape(proccessed_state, [84, 84])
                self.state.clear()
                self.state += 4 * [proccessed_state]
                self.state_buffer.append(self.state[:])
            else:
                # append the last stop state to state buffer
                self.state_buffer.append(self.state[:])

            # interact with the environment for t_max steps or till terminal step
            for t in range(self.t_max):
                # select action
                # if threading.current_thread().name == "Worker_2":
                #    self.env.render()
                c_lives = self.env.env.ale.lives()
                action_prob, value = sess.run([self.w_network.policy, self.w_network.value],
                                              {self.w_network.state_u: np.reshape(self.state, [1, 84, 84, 4])})
                action = np.random.choice(
                    np.arange(self.num_actions), p=action_prob)
                # pass action
                observation, reward, self.done, info = self.env.step(
                    action)
                lives = self.env.env.ale.lives()
                # process the new state
                proccessed_state = sess.run([self.w_network.proc_state], {
                                            self.w_network.observation: observation})
                proccessed_state = np.reshape(proccessed_state, [84, 84])

                # reward clipping
                if reward > 0:
                    reward = 1
                elif reward < 0:
                    reward = -1

                # pop's the item for a given index
                self.state.pop(0)
                self.state.append(proccessed_state)
                self.value_state.append(np.reshape(value, [1]))
                self.reward.append(reward)
                self.episode_reward += reward
                self.action.append(action)
                self.state_buffer.append(self.state[:])

                self.steps_worker += 1
                local_t = next(self.local_counter)
                global_t = next(self.global_counter)

                if local_t % 100 == 0:
                    tf.logging.info("{}: local Step {}, global step {}".format(
                        self.id, local_t, global_t))

                # return the value of the last state
                if self.done or (c_lives != lives):
                    episode_count = next(self.episode_counter)
                    count += 1
                    '''change name'''
                    if threading.current_thread().name == "Worker_{}1".format(workers_list[0]):
                        summaries, global_step = sess.run(
                            [self.w_network.summaries,
                             self.global_step], feed_dict={self.w_network.episode_reward: self.episode_reward}
                        )
                        self.writer.add_summary(summaries, global_step)
                        self.writer.flush()
                        if count % 5 == 0:
                            print("Global Episode Count:", episode_count)
                            print("Global Steps", global_t)
                    self.episode_reward = 0
                    self.reward.append(0)
                    break
                elif t == (self.t_max - 1):
                    value = sess.run([self.w_network.value],
                                     {self.w_network.state_u: np.reshape(self.state, [1, 84, 84, 4])})
                    self.reward.append(np.reshape(value, [1]))

            self.r_return = [self.reward[len(self.reward) - 1]]
            num_steps = len(self.state_buffer)

            # number of steps may not always be equal to t_max
            for t in range(num_steps - 1):
                self.r_return.append(
                    self.reward[len(self.reward) - 2 - t] + self.discount * self.r_return[t])

            # removing the value of the last state
            self.r_return.pop(0)
            # reversing the return list to match the indexes of other lists: index order (t+4 -> t)
            self.r_return.reverse()
            # remove the last state from the state buffer, as it will not be used
            self.state_buffer.pop(num_steps - 1)
            num_steps -= 1

            # calculating advantage
            advantage = list(
                map(operator.sub, self.r_return, self.value_state))
            learning_rate = self.anneal_learning_rate(global_t)

            # popping the value reward from reward buffer
            feed_dict = {
                self.w_network.advantage: np.reshape(advantage, [1, num_steps]),
                self.w_network.actions: np.reshape(self.action, [1, num_steps]),
                self.w_network.state_u: np.reshape(self.state_buffer, [num_steps, 84, 84, 4]),
                self.w_network.reward: np.reshape(self.r_return, [1, num_steps]),
                self.w_network.learning_rate: learning_rate,
                self.global_network.learning_rate: learning_rate
            }

            # calculating and applying the gradients
            _ = sess.run([self.grad_apply], feed_dict)

            self.state_buffer.clear()
            self.reward.clear()
            self.value_state.clear()
            self.action.clear()
            self.r_return.clear()

            if global_t > self.max_global_time_step:
                coord.request_stop()
                return

            if threading.current_thread().name == "Worker_1":
                if count % 50 == 0 and count != 0:
                    saver.save(sess, CHECKPOINT_DIR +
                               '/model-' + str(episode_count) + '.cptk')
                    print("Saved Model")

    def anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (
            self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate


def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)


if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    game = "Qbert-v0"
    #game = "Pong-v0"
    #game = "Breakout-v0"
    #game = "CartPole-v0"
    # observation/state array shape: (210,160,3)
    # every action is performed for a duration of k frames, where k
    # is sampled from {2,3,4}
    # action space: 6 for Qbert
    # 0,1,2,3,4,5 : possible actions

    env = gym.make(game)
    num_actions = env.action_space.n
    env.close()

    #num_cores = multiprocessing.cpu_count()
    num_cores = 1
    t_max = 5
    print("Num Cores", num_cores)
    gamma = 0.99
    DIR = "/A3C/"
    max_global_time_step = 16000  # 320 * 1000000
    alpha_low = 1e-4
    alpha_high = 1e-2
    alpha_log_rate = 0.4226
    clip_norm = 40.0
    global_counter = itertools.count()

    tf.flags.DEFINE_string("model_dir", "experiments/exp1",
                           "Directory to write Tensorboard summaries and videos to.")
    FLAGS = tf.flags.FLAGS
    MODEL_DIR = FLAGS.model_dir

    CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"))
    initial_learning_rate = log_uniform(alpha_low, alpha_high, alpha_log_rate)
    #initial_learning_rate = 0.00025

    with tf.device("/cpu:0"):
        tf.reset_default_graph()

        global_step = tf.Variable(0, name="global_step", trainable=False)

        with tf.variable_scope("global"):
            global_network = GlobalPolicyValueNetwork(num_actions, "global")

        global_counter = itertools.count()
        episode_counter = itertools.count()

        workers = []
        num_cores = 1
        for i in range(num_cores):
            worke = Worker(game, "worker_{}{}".format(FLAGS.task_index, i + 1), t_max, num_actions, global_network, gamma, writer,
                           initial_learning_rate, max_global_time_step, clip_norm, global_counter, episode_counter)
            workers.append(worke)
        saver = tf.train.Saver(max_to_keep=10)
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(worker_n == 0)) as mon_sess:
        while not mon_sess.should_stop():
            coord = tf.train.Coordinator()

            latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
            if latest_checkpoint:
                print("Loading model checkpoint: {}".format(latest_checkpoint))
                saver.restore(sess, latest_checkpoint)

            writer.add_graph(graph=mon_sess.graph)

            threads = []
            i = 1
            for worker in workers:
                work = lambda worker=worker: worker.play(
                    coord, mon_sess, saver, CHECKPOINT_DIR)
                t = threading.Thread(name="Worker_{}{}".format(
                    FLAGS.task_index, i), target=work)
                i = i + 1
                threads.append(t)
                t.start()

        coord.join(threads)
