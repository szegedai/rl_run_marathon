import pickle
import gym
import numpy as np
import random
from policy_dynamic import Policy
from value_function import NNValueFunction
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import tensorflow as tf


all_steps = 0
max_it = 125
ep_it = 160
model_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../model'))


def init_gym(environment, seed):
    env = gym.make(environment)
    env.action_space.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def run_episode(env, policy, scaler):
    global all_steps
    global max_it
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0
    offset[-1] = 0.0
    i = 0
    counter = 0
    obs = obs[0]
    while env.is_healthy and i < max_it:
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        action = action[0]
        obs, reward, _, _, _ = env.step(action)
        all_steps += 1
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3
        i += 1
        if i == max_it:
            counter = 1

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float32), np.concatenate(unscaled_obs), counter)


def run_policy(env, policy, scaler, logger, episode, save_steps):
    global model_path
    global all_steps
    global ep_it
    global max_it

    total_steps = 0
    trajectories = []
    all_counter = 0
    for e in range(ep_it):
        observes, actions, rewards, unscaled_obs, counter = run_episode(env, policy, scaler)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
        all_counter += counter

    if max_it == 2000 and all_counter > 7:
        max_it = 4000
        ep_it = 5
    if max_it == 1000 and all_counter > 13:
        max_it = 2000
        ep_it = 10
    if max_it == 500 and all_counter > 27:
        max_it = 1000
        ep_it = 20
    if max_it == 250 and all_counter > 53:
        max_it = 500
        ep_it = 40
    if max_it == 125 and all_counter > 107:
        max_it = 250
        ep_it = 80

    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)
    scalar_data = {"vars": scaler.vars, "means": scaler.means, "m": scaler.m}
    episode += ep_it
    if(policy.all_steps_remainder < all_steps//save_steps):
        if not os.path.exists(model_path + '/' + str(all_steps) + '/info'):
            os.makedirs(model_path + '/' + str(all_steps) + '/info')
        with open(model_path + '/' + str(all_steps) + "/info/scalar.pkl", "wb") as f:
            pickle.dump(scalar_data, f)

    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})

    return trajectories


def discount(x, gamma):
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    for trajectory in trajectories:
        if gamma < 0.999:
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam):
    for trajectory in trajectories:
        if gamma < 0.999:
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode
                })


def main(gamma, lam, kl_targ, seed, environment, total_training_steps, save_steps):
    global model_path
    global all_steps

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    model_dirs = os.listdir(model_path)
    if(model_dirs == []):
        model_folder = '001'
        model_path = model_path + '/001'
        os.makedirs(model_path)
    else:
        model_dirs.sort()
        dir_number = "{:03d}".format(int(model_dirs[-1]) + 1)
        model_folder = str(dir_number)
        model_path = model_path + '/' + str(dir_number)
        os.makedirs(model_path)

    env, obs_dim, act_dim = init_gym(environment, seed)
    obs_dim += 1

    now = datetime.now().strftime("%Y-%m-%d_%H" + 'h' + "_%M" + 'm' + "_%S" + 's' + '--' + model_folder)
    logger = Logger(logname=environment, now=now)
    episode = 0
    scaler = Scaler(obs_dim)
    val_func = NNValueFunction(obs_dim, model_path, seed)
    policy = Policy(obs_dim, act_dim, kl_targ, model_path, seed, save_steps)

    run_policy(env, policy, scaler, logger, episode, save_steps)

    while all_steps < total_training_steps:
        trajectories = run_policy(env, policy, scaler, logger, episode, save_steps)
        episode += len(trajectories)
        add_value(trajectories, val_func)
        add_disc_sum_rew(trajectories, gamma)
        add_gae(trajectories, gamma, lam)
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)
        policy.update(observes, actions, advantages, logger, all_steps)
        val_func.fit(observes, disc_sum_rew, logger)
        logger.write(display=True)
        if all_steps > total_training_steps:
            policy.save_policy(all_steps)
    logger.close()
    policy.close_sess()
    val_func.close_sess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('--seed', type=int, help='Set seed', default=0)
    parser.add_argument('--environment', type=str, help='Openai GYM Mujoco Environment', default=None)
    parser.add_argument('--total_training_steps', type=int, help='Total number of steps allowed during the entire training process', default=None)
    parser.add_argument('--save_steps', type=int, help='Number of steps after which the models are saved during training', default=None)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    main(**vars(args))
