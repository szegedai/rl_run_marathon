import numpy as np
import tensorflow as tf


class Policy(object):
    def __init__(self, obs_dim, act_dim, model_path):
        self.model_path = model_path

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        self.g = tf.compat.v1.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._sample()
            self.init = tf.compat.v1.global_variables_initializer()
            self.saver = tf.compat.v1.train.Saver()

    def _placeholders(self):
        self.obs_ph = tf.compat.v1.placeholder(tf.compat.v1.float32, (None, self.obs_dim), 'obs')

    def _policy_nn(self):
        hid1_size = self.obs_dim * 10
        hid3_size = self.act_dim * 10
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        out = tf.compat.v1.layers.dense(self.obs_ph, hid1_size, tf.compat.v1.tanh,
                              kernel_initializer=tf.compat.v1.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)), name="h1")
        out = tf.compat.v1.layers.dense(out, hid2_size, tf.compat.v1.tanh,
                              kernel_initializer=tf.compat.v1.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid1_size)), name="h2")
        out = tf.compat.v1.layers.dense(out, hid3_size, tf.compat.v1.tanh,
                              kernel_initializer=tf.compat.v1.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid2_size)), name="h3")
        self.means = tf.compat.v1.layers.dense(out, self.act_dim,
                                     kernel_initializer=tf.compat.v1.random_normal_initializer(
                                         stddev=np.sqrt(1 / hid3_size)), name="means")

    def _sample(self):
        self.sampled_act=self.means

    def _init_session(self):
        with open(self.model_path + '/info/episodes.txt') as f:
            self.episodes = int(f.readlines()[0])
        with self.g.as_default():
            self.sess = tf.compat.v1.Session()
            new_saver = tf.compat.v1.train.import_meta_graph(self.model_path + '/model-' + str(self.episodes) + '.meta')
            new_saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))

    def sample(self, obs):
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(self.sampled_act, feed_dict=feed_dict)
