import os
import random
import shutil

import numpy as np
import pandas as pd


class NeuralNet:
    def __init__(self):
        self.layers = []
        self.activations = []

    def add_layer(self, weights, activation=None):
        self.layers.append(weights)
        self.activations.append(activation)
    @staticmethod
    def get_models_dir():
        return os.path.join(os.path.dirname(__file__), "saved_models")
        
    def save_model(self, model_name):
        dir_path = self.get_models_dir()
        model_dir = os.path.join(dir_path, model_name)
        if model_name in os.listdir(dir_path):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir, exist_ok=True)

        for i, layer in enumerate(self.layers):
            layer_number = str(i) if i >= 10 else "0" + str(i)
            layer_name = f"layer_{layer_number}.csv"
            save_file = os.path.join(model_dir, layer_name)
            df = pd.DataFrame(layer)
            df.to_csv(save_file, sep=",", header=None)

    def load_model(self, model_name):
        self.layers = []
        self.activations = []
        dir_path = os.path.join(os.path.dirname(__file__), "saved_models")
        model_dir = os.path.join(dir_path, model_name)
        model_files = os.listdir(model_dir)

        layer_files = list(filter(lambda x: x.startswith("layer"), model_files))
        layer_files.sort()
        for layer_file in layer_files:
            layer = pd.read_csv(
                os.path.join(model_dir, layer_file), header=None
            )
            self.layers.append(layer.values[:, 1:])
            self.activations.append(None)

    def run(self, input_vector):
        for i, layer in enumerate(self.layers):
            input_vector = np.matmul(
                input_vector.transpose(), layer
            ).transpose()
            if self.activations[i] is not None:
                input_vector = self.activations[i](input_vector)
        return input_vector


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def build_random_network(layer_shape):
    nn = NeuralNet()
    for i in range(len(layer_shape) - 1):
        nn.add_layer(
            np.random.random(size=(layer_shape[i][0], layer_shape[i + 1][0]))
            - 0.5,
            layer_shape[i + 1][1],
        )
    return nn


def choice_function(network):
    def to_ret(observation):
        args = network.run(observation)
        action = np.argmax(args)
        return int(action)

    return to_ret


def choose_breed(generation):
    """
    Use to breed a new generation from an existing generation.
    """
    to_ret = []
    fitnesses = [g[0] for g in generation]
    fitnesses = [x - min(fitnesses) for x in fitnesses]
    total_fitnesses = sum(fitnesses)
    fitness_probability = fitnesses.copy()
    fitness_probability = [x / total_fitnesses for x in fitness_probability]
    for _ in range(200):
        breed_choices = []
        for _ in range(2):
            test_num = np.random.random()
            prob = 0
            for i, p in enumerate(fitness_probability):
                prob += p
                if test_num < prob:
                    breed_choices.append(generation[i][1])
                    break
        to_ret.append(breed_choices)
    return to_ret


def breed_networks(network1, network2, generation, activations):
    """Assume of same shape for now"""
    to_ret = NeuralNet()
    to_ret.activations = activations
    for i in range(len(network1.layers)):
        layer1, layer2 = network1.layers[i], network2.layers[i]
        # mask = np.random.random(layer1.shape)
        # mask = mask > 0.5
        # layer1[mask] = layer2[mask]

        to_ret.add_layer(random.choice((layer1, layer2)))
    return to_ret


if __name__ == "__main__":
    random_network = build_random_network(
        (
            (4, None),
            (8, None),
            (4, None),
        )
    )
    layer_zero = random_network.layers[0]
    random_network.save_model("test")
    random_network.load_model("test")
