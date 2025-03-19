import numpy as np
from policy import Policy
from scaler import Scaler
import os
import argparse
import pandas as pd
import gym
import tensorflow as tf


def init_gym(environment):
    wrappedEnv = gym.make(environment)
    obs_dim = wrappedEnv.env.observation_space.shape[0]
    act_dim = wrappedEnv.env.action_space.shape[0]
    return wrappedEnv, obs_dim, act_dim


def run_episode(policy, scaler, env, seed):
    obs = env.reset(seed=seed)
    step = 0.0
    n_step = 0
    scale, offset = scaler.get()
    scale[-1] = 1.0
    offset[-1] = 0.0
    rewards = 0
    obs = obs[0]
    while env.is_healthy:
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)
        obs = (obs - offset) * scale
        action = policy.sample(obs).reshape((1, -1)).astype(np.float64)
        action = action[0]
        obs, _, _, _, info = env.step(action)
        rewards += info['x_velocity']
        step += 1e-3
        n_step += 1
    return n_step, rewards


def run_policy(policy, scaler, env, seed):
    return run_episode(policy, scaler, env, seed)


def main(models, excel_name, environment):
    datas = {"Model": [], "Steps": [], "Reward Divided by Steps": [], "Rewards": [], "Seed": []}
    models = models.split()
    models_names = []
    for model in models:
        models_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../model/' + model))
        model_temps_paths = os.listdir(models_path)
        model_temps_paths = [model + '/' + sub_model for sub_model in model_temps_paths]
        models_names.extend(model_temps_paths)

    for seed in range(0, 10):
        model_array = models_names

        for model_name in model_array:
            model_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../model/' + model_name))
            env, obs_dim, act_dim = init_gym(environment)
            obs_dim += 1
            model_subpath = model_name.split('/')[1]
            scaler = Scaler(model_path)
            policy = Policy(obs_dim, act_dim, model_path, model_subpath)
            steps, rewards = run_policy(policy, scaler, env, seed)
            datas["Model"].append(model_name)
            datas["Steps"].append(steps)
            datas["Reward Divided by Steps"].append(rewards / steps)
            datas["Rewards"].append(rewards)
            datas["Seed"].append(seed)

        if not os.path.exists('results'):
            os.makedirs('results')
        writer = pd.ExcelWriter('results/' + excel_name + '.xlsx', engine = 'xlsxwriter')

        datas_df = pd.DataFrame(datas)
        datas_df.to_excel(writer, sheet_name="Sheet1")

    writer.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('models', type=str, help='OpenAI Gym model folder')
    parser.add_argument('-en', '--excel_name', type=str, help='Output Excel Name', required=True)
    parser.add_argument('-env', '--environment', type=str, help='Env Name (like Hopper-v4 or Walker2d-v4)', required=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    args = parser.parse_args()
    main(**vars(args))
