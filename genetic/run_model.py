import os
import re

from genetic.genetic_main_loop import run_sim
from neural_network import NeuralNet, choice_function


def run_models():
    models = os.listdir(NeuralNet.get_models_dir())
    models = [x for x in models if re.match(r"\d+_2", x)]
    models.sort(
        key=lambda x: os.path.getmtime(
            os.path.join(
                os.path.dirname(__file__), NeuralNet.get_models_dir(), x
            )
        )
    )

    models = models[0:60]
    max_average = 0
    max_model = None
    num_models = 30

    for model_name in models:

        model_average = 0
        model = NeuralNet()
        model.load_model(model_name)
        model = choice_function(model)

        for _ in range(num_models):
            reward = run_sim(model, None, max_steps=5000)
            print(f"reward for sim is: {reward}")
            model_average += reward
        model_average = model_average / num_models
        max_average = max(max_average, model_average)

        if max_average == model_average:
            max_model = model_name

        print(f"average reward for model {model_name} is {model_average}")
    print(f"max model was {max_model} with a score of {max_average}")


if __name__ == "__main__":
    run_models()
