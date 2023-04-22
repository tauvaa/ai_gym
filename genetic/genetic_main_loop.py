import itertools as it
import random

import gymnasium as gym
import numpy as np

from genetic.neural_network import (breed_networks, build_random_network,
                                    choice_function, choose_breed, sigmoid)


def random_model(observation):
    return np.random.randint(0, 4)


def run_sim(model, render_mode):

    env = gym.make("LunarLander-v2", render_mode=render_mode)
    observation, _ = env.reset()
    done = False
    total_reward = 0
    step_number = 0
    while not done:
        action = model(observation)

        observation, reward, done, truc, _ = env.step(action)
        step_number += 1
        total_reward += reward
        if total_reward < -300 or step_number > 500:
            done = True

    env.close()
    return total_reward


if __name__ == "__main__":
    network_shape = (
        (8, None),
        (250, None),
        # (250, sigmoid),
        # (250, sigmoid),
        # (250, sigmoid),
        # (250, sigmoid),
        (4, None),
    )
    gen_zero = [build_random_network(network_shape) for _ in range(100)]
    all_networks = []
    gen_counter = 0
    for i, network in enumerate(gen_zero):
        choice = choice_function(network)
        average_reward = 0
        for _ in range(5):
            total_reward = run_sim(choice, None)
            average_reward += total_reward
        average_reward /= 5
        all_networks.append((average_reward, network))

    while True:
        all_networks.sort(key=lambda x: x[0], reverse=True)
        breed_nets = all_networks[0:15]
        breed_networks_average = sum([x[0] for x in breed_nets]) / len(
            breed_nets
        )
        best_network = breed_nets[0]
        print(
            f"generation: {gen_counter}, average: {breed_networks_average}, best: {best_network[0]}"
        )
        gen_networks = choose_breed(all_networks)

        gen_networks = [
            breed_networks(
                *x,
                gen_counter,
                activations=[x[1] for x in network_shape[1:]],
            )
            for x in gen_networks
        ]
        gen_networks += [
            breed_networks(
                best_network[1],
                build_random_network(network_shape),
                gen_counter,
                activations=[x[1] for x in network_shape[1:]],
            )
            for _ in range(5)
        ]
        gen_networks += [
            breed_networks(
                breed_nets[1][1],
                build_random_network(network_shape),
                gen_counter,
                activations=[x[1] for x in network_shape[1:]],
            )
            for _ in range(5)
        ]
        gen_networks += [build_random_network(network_shape) for _ in range(10)]

        num_to_keep = int(0.1 * len(all_networks))
        best_networks = list(map(lambda x: x[1], all_networks[0:num_to_keep]))
        gen_networks += best_networks
        all_networks = []


        for i, network in enumerate(gen_networks):
            choice = choice_function(network)
            average_reward = 0
            sim_mode = "human" if gen_counter % 10 == 0 and i == 0 else None
            for _ in range(5):
                total_reward = run_sim(choice, sim_mode)
                average_reward += total_reward
            average_reward /= 5
            all_networks.append((average_reward, network))
        print(f"gen network length: {len(gen_networks)}")
        gen_counter += 1
