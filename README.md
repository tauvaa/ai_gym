# AI Gym
## Description
This project is to build reinforcement learning applications to solve the atari games provided by [AI gym](https://github.com/openai/gym).  The algorithms used are Q-learning and genetic.
## Environment
To set up the environment first clone the repor and naviate to the root.  Then create a python virtual environment and install the requirements.
```
python3 -m venv venv
pip3 install -r requirements_tensorflow.txt
```
## Q-Learning
The Q-learning algorithm uses an [agent](https://github.com/tauvaa/ai_gym/blob/main/deep_q_learning/agent.py) and [replay buffer](https://github.com/tauvaa/ai_gym/blob/main/deep_q_learning/replay_buffer.py) to train a Neural Network capable of solving the lunar landing problem.  The network is trained using tensorf flow.  To begin training run

`python3 ./main_loop.py`

This will begin training, showing updates every 5 training cycles.  It will also save your best networks in the models/tensorflow directory.  After aprrox 100 iterations you should be getting scores of around 200. Once you have a good sample of 200+ models, you can run the [test_models](https://github.com/tauvaa/ai_gym/blob/main/ai_utils/test_models.py) script (`PYTHONPATH=./ python3 ai_utils/test_models.py`).  This willl run the 190+ models and take an average over 3, saving the best models in a models_info.json file.  Once this is done you can run the [choose_models](https://github.com/tauvaa/ai_gym/blob/main/ai_utils/choose_model.py) script which will run the five best models in human mode, so you can see the lander land.

## Genetic Algorithm
