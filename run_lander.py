import os

import gymnasium as gym
import numpy as np
import tensorflow as tf


def get_action(model, observation):
    observation = np.array([observation])
    pred = model.predict(observation)
    return np.argmax(pred[0])


def run_lander():
    model = tf.keras.saving.load_model("models/tensorflow/example-lunar/")
    env = gym.make("LunarLander-v2", render_mode="human")
    for _ in range(10):
        total_reward = 0

        observation, _ = env.reset()
        while True:
            action = get_action(model, observation)
            observation, reward, done, truc, _ = env.step(action)
            total_reward += reward
            if done or total_reward < -100:
                print(f"total reward is: {total_reward}")
                break


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    run_lander()
