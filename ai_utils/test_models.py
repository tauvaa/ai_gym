import json
import os

import gymnasium as gym
import numpy as np
import tensorflow as tf

model_folder = os.path.join(os.path.split(os.path.dirname(__file__))[0], "models", "tensorflow")
print(model_folder)

models = [
    os.path.join(model_folder, x)
    for x in os.listdir(model_folder)
    if int(x.split("-")[-1]) > 190
]


def make_choice(observation, model):
    observation = np.array([observation])
    pred = model.predict(observation)
    return np.argmax(pred[0])


def run_sim(model, num_sims, render_mode=None):
    for _ in range(num_sims):
        env = gym.make("LunarLander-v2", render_mode=render_mode)
        observation, _ = env.reset()
        total_reward = 0
        all_totals = []
        while True:
            action = make_choice(observation, model)
            observation, reward, done, trun, _ = env.step(action)
            total_reward += reward
            if done or total_reward < -100:
                all_totals.append(total_reward)
                break
    return np.array(all_totals).mean()


if __name__ == "__main__":

    tf.compat.v1.disable_eager_execution()
    model_info = {}
    for m in models:
        model = tf.keras.saving.load_model(m)
        model_info[m] = run_sim(model, 3)
        print(model_info)
    with open("model_info.json", "w+") as f:
        json.dump(model_info, f)
