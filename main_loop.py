import gymnasium as gym
import numpy as np
import tensorflow

from deep_q_learning.agent import Agent

SAVE_VALUE = 200
STOP_VALUE = -200


def main():
    # for updating on high enough reward
    back_track_value = 0
    back_track_counter = 0
    back_track_max = 250
    back_track_num = 5
    agent = Agent(input_dims=(8,))
    for episode in range(500):
        print(f"running episode: {episode}")
        total_reward = 0
        observation, _ = env.reset()
        update_indexes = []
        render_mode = "human" if episode % 5 == 0 else None
        env = gym.make("LunarLander-v2", render_mode=render_mode)

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
            if done or total_reward < STOP_VALUE:
                # back track 
                if total_reward > back_track_value:
                    update_indexes = np.array(update_indexes, dtype=np.int32)
                    agent.memory.rewards[update_indexes] += 5
                    back_track_counter += 1
                    if back_track_counter > back_track_num:
                        back_track_counter = 0
                        back_track_value += 25
                        back_track_value = min(back_track_value, back_track_max)

                if total_reward > SAVE_VALUE:
                    tensorflow.keras.saving.save_model(
                        agent.model,
                        f"./models/tensorflow/lunar-{episode}-{int(total_reward)}",
                    )
                    agent.model = tensorflow.keras.saving.load_model(
                        f"./models/tensorflow/lunar-{episode}-{int(total_reward)}",
                    )
                print(f"episode reward was: {total_reward}")
                break


if __name__ == "__main__":
    tensorflow.compat.v1.disable_eager_execution()
    main()
