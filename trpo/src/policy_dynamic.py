import numpy as np
import tensorflow as tf
import os


class Policy(object):
    def __init__(self, obs_dim, act_dim, kl_targ, model_path, seed, divider):
        self.divider = divider
        self.seed = seed
        self.model_path = model_path
        self.graph_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../graph'))
        self.episodes = 0
        self.all_steps_remainder = 0
        self.all_steps = 0
        self.beta = 1.0
        self.lr_multiplier = 1.0
        self.lr = None
        self.kl_targ = kl_targ
        self.eta = 50
        self.epochs = 20
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        self.g = tf.compat.v1.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._logprob()
            self._kl_entropy()
            self._sample()
            self._loss_train_op()
            self.init = tf.compat.v1.global_variables_initializer()
            self.saver = tf.compat.v1.train.Saver(max_to_keep=1000) 

    def _placeholders(self):
        self.obs_ph = tf.compat.v1.placeholder(tf.compat.v1.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.compat.v1.placeholder(tf.compat.v1.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.compat.v1.placeholder(tf.compat.v1.float32, (None,), 'advantages')

        self.beta_ph = tf.compat.v1.placeholder(tf.compat.v1.float32, (), 'beta')
        self.eta_ph = tf.compat.v1.placeholder(tf.compat.v1.float32, (), 'eta')

        self.lr_ph = tf.compat.v1.placeholder(tf.compat.v1.float32, (), 'eta')

        self.old_log_vars_ph = tf.compat.v1.placeholder(tf.compat.v1.float32, (self.act_dim,), 'old_log_vars')
        self.old_means_ph = tf.compat.v1.placeholder(tf.compat.v1.float32, (None, self.act_dim), 'old_means')

    def _policy_nn(self):
        hid1_size = self.obs_dim * 10
        hid3_size = self.act_dim * 10
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        self.lr = 9e-4 / np.sqrt(hid2_size)

        out = tf.compat.v1.layers.dense(self.obs_ph, hid1_size, tf.compat.v1.tanh,
                              kernel_initializer=tf.compat.v1.random_normal_initializer(
                                  seed=self.seed, stddev=np.sqrt(1 / self.obs_dim)), name="h1")
        out = tf.compat.v1.layers.dense(out, hid2_size, tf.compat.v1.tanh,
                              kernel_initializer=tf.compat.v1.random_normal_initializer(
                                  seed=self.seed, stddev=np.sqrt(1 / hid1_size)), name="h2")
        out = tf.compat.v1.layers.dense(out, hid3_size, tf.compat.v1.tanh,
                              kernel_initializer=tf.compat.v1.random_normal_initializer(
                                  seed=self.seed, stddev=np.sqrt(1 / hid2_size)), name="h3")
        self.means = tf.compat.v1.layers.dense(out, self.act_dim,
                                     kernel_initializer=tf.compat.v1.random_normal_initializer(
                                         seed=self.seed, stddev=np.sqrt(1 / hid3_size)), name="means")

        logvar_speed = (10 * hid3_size) // 48
        log_vars = tf.compat.v1.get_variable('logvars', (logvar_speed, self.act_dim), tf.compat.v1.float32,
                                   tf.compat.v1.constant_initializer(0.0))
        self.log_vars = tf.compat.v1.reduce_sum(log_vars, axis=0) - 1.0

        print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}, logvar_speed: {}'
              .format(hid1_size, hid2_size, hid3_size, self.lr, logvar_speed))

    def _logprob(self):
        logp = -0.5 * tf.compat.v1.reduce_sum(self.log_vars)
        logp += -0.5 * tf.compat.v1.reduce_sum(tf.compat.v1.square(self.act_ph - self.means) /
                                     tf.compat.v1.exp(self.log_vars), axis=1)
        self.logp = logp

        logp_old = -0.5 * tf.compat.v1.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.compat.v1.reduce_sum(tf.compat.v1.square(self.act_ph - self.old_means_ph) /
                                         tf.compat.v1.exp(self.old_log_vars_ph), axis=1)
        self.logp_old = logp_old

    def _kl_entropy(self):
        log_det_cov_old = tf.compat.v1.reduce_sum(self.old_log_vars_ph)
        log_det_cov_new = tf.compat.v1.reduce_sum(self.log_vars)
        tr_old_new = tf.compat.v1.reduce_sum(tf.compat.v1.exp(self.old_log_vars_ph - self.log_vars))

        self.kl = 0.5 * tf.compat.v1.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.compat.v1.reduce_sum(tf.compat.v1.square(self.means - self.old_means_ph) /
                                                     tf.compat.v1.exp(self.log_vars), axis=1) -
                                       self.act_dim)
        self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                              tf.compat.v1.reduce_sum(self.log_vars))

    def _sample(self):
        self.sampled_act = self.means + tf.math.exp(self.log_vars / 2.0) * tf.random.normal(shape=(self.act_dim,), seed=self.seed)

    def _loss_train_op(self):
        loss1 = -tf.compat.v1.reduce_mean(self.advantages_ph *
                                tf.compat.v1.exp(self.logp - self.logp_old))
        loss2 = tf.compat.v1.reduce_mean(self.beta_ph * self.kl)
        loss3 = self.eta_ph * tf.compat.v1.square(tf.compat.v1.maximum(0.0, self.kl - 2.0 * self.kl_targ))
        self.loss = loss1 + loss2 + loss3
        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)

    def _init_session(self):
        self.sess = tf.compat.v1.Session(graph=self.g)
        self.sess.run(self.init)

    def sample(self, obs):
        feed_dict = {self.obs_ph: obs}

        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages, logger, all_steps):
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars],
                                                  feed_dict={self.obs_ph: observes})

        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.lr_ph: self.lr * self.lr_multiplier,
                     self.old_log_vars_ph: old_log_vars_np,
                     self.old_means_ph: old_means_np}

        loss, kl, entropy = 0, 0, 0
        for _ in range(self.epochs):
            self.sess.run(self.train_op, feed_dict)
            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_targ * 4:
                break
        if kl > self.kl_targ * 2:
            self.beta = np.minimum(35, 1.5 * self.beta)
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)
            if self.beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

        self.all_steps = all_steps
        if(self.all_steps_remainder < all_steps//self.divider):
            self.all_steps_remainder = all_steps//self.divider
            with open(self.model_path + '/' + str(all_steps) + '/info/episodes.txt', 'w') as f:
                f.write(str(all_steps))
            self.save_policy()

        logger.log({'PolicyLoss': loss,
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.beta,
                    '_lr_multiplier': self.lr_multiplier})

    def save_policy(self):
        self.saver.save(self.sess, self.model_path + '/' + str(self.all_steps) + '/model', global_step=self.all_steps)

    def close_sess(self):
        self.sess.close()
