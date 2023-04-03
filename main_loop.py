import gymnasium as gym
import numpy as np
import tensorflow

from deep_q_learning.agent import Agent
SAVE_VALUE = 200


def main():
    back_track_value = 0
    back_track_counter = 0
    back_track_num = 5
    agent = Agent(input_dims=(8,))
    env = gym.make("LunarLander-v2", render_mode="human")
    for episode in range(500):
        print(f"running episode: {episode}")
        total_reward = 0
        observation, _ = env.reset()
        update_indexes = []

        while True:
            action = agent.chose_action(observation)
            next_observation, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if action == 0:
                reward += 1
            update_indexes.append(
                agent.memory.buf_index % agent.memory.max_memory
            )
            agent.memory.store_memory(
                observation, next_observation, action, reward, done
            )
            if episode % 5 != 0:
                agent.learn()
            observation = next_observation
            if done or total_reward < -200:
                if total_reward > 100:
                    update_indexes = np.array(update_indexes, dtype=np.int32)
                    agent.memory.rewards[update_indexes] += 5
                if total_reward > 150:
                    tensorflow.keras.saving.save_model(
                        agent.model,
                        f"./models/tensorflow/lunar-{episode}-{int(total_reward)}",
                    )
                    print(f"total reward for episode was: {total_reward}")
                    agent.model = tensorflow.keras.saving.load_model(
                        f"./models/tensorflow/lunar-{episode}-{int(total_reward)}",
                    )
                print(f"episode reward was: {total_reward}")
                break


if __name__ == "__main__":
    tensorflow.compat.v1.disable_eager_execution()
    main()
