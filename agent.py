import numpy as np
import json
import tensorflow as tf

from replay_buffer import ReplayBuffer


def get_model():
    model = tf.keras.Sequential(
        [
            # tf.keras.layers.Dense(200, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(50, activation=tf.keras.activations.sigmoid),
            tf.keras.layers.Dense(50, activation=tf.keras.activations.sigmoid),
            tf.keras.layers.Dense(4),
        ]
    )

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer="adam")
    return model


class Agent:
    def __init__(self, input_dims, model=None) -> None:
        self.model = get_model()
        self.memory = ReplayBuffer(input_dims)
        self.gamma = 0.95
        self.epsilon = 0.9
        self.out_dim = 4
        self.epsilon_stop = 0.05

    def chose_action(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.out_dim))
        observation = np.array([observation])
        pred = self.model.predict(observation)
        action = np.argmax(pred[0])
        return action

    def learn(self):
        sample_size = 128
        if self.memory.buf_index < sample_size:
            return
        (
            observations,
            next_observations,
            actions,
            rewards,
            terminated,
        ) = self.memory.sample_buffer(sample_size)
        pred_vals = self.model.predict(observations)
        next_pred_vals = self.model.predict(next_observations)
        targets = np.copy(pred_vals)
        targets[
            np.arange(sample_size, dtype=np.int32), actions
        ] = rewards + self.gamma * (np.max(next_pred_vals, axis=1) * terminated)

        self.model.train_on_batch(observations, targets)
        self.epsilon -= 0.01 if self.epsilon > self.epsilon_stop else 0
