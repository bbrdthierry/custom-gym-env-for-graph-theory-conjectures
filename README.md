# Custom OpenAI Gym Environment for Graph Theory Conjectures
This custom OpenAI Gym Environment was originally developed to contribute to the 99-vertex Conway graph problem. But it is a more general reinforcement learning solution to find counterexamples to graph theory conjectures, based on the "Constructions in combinatorics via neural networks" paper by A Z Wagner.

Conway's 99-graph problem:
https://en.wikipedia.org/wiki/Conway%27s_99-graph_problem

Inspired by the demo code associated with "Constructions in combinatorics via neural networks" by A Z Wagner paper:
https://arxiv.org/abs/2104.14516

Repository's paper, by A Z Wagner:
https://github.com/zawagner22/cross-entropy-for-combinatorics

## Software requirements

- Python version 3.6.3
- Tensorflow version 1.14.0
- Keras version 2.3.1
- Gym version 0.21.0
- Stable Baselines version 2.10.0

Python and minor packages are also available in requirements.txt, which can be used when creating a Python/Conda virtual environment.

## Demos

The custom OpenAI Gym Environment is developed inside the **cge-custom_env.py** file. In this file, the conjecture is represented by the *custom reward function*: only modify this one when you want to find another counter exemple to a conjecture. By default, this reward function is designed to represent a tree graph.

**cge-model.py** file contains the reinforcement learning model based on Stable Baseline API. Check their RL model list in https://stable-baselines.readthedocs.io/en/master/guide/algos.html. Using Stable Baseline API allows us to easily find the best RL model that matches the best a conjecture problem. When you have found the right model, feel free to reimplement it with this custom Gym environment without Stable-Baseline to get more model tuning solutions!

**cge-playground.py** file is a toy module for the Gym environment. You can run it to play with the environment, and see how the latter gives outputs. *Not needed to find counter examples*.

**cge-check_env.py** file is designed to check the compatibility between Gym environment and Stable Baseline, as the RL environment is a custom one. *Not needed to find counter examples*.

## Installation and usage

Install the right versions of modules cited in **Software Requirements** section.
The main files are **cge-model.py** and **cge-custom_env.py**. Inside **cge_model.py**, change the argument `N_vertices` value by the number of vertices that you want, by default it is set on 17. Next in **cge-custom_env.py** file, change the reward function that suits your problem under the `custom_reward` function. The input to this function is a networkx graph object, which is automatically designed when specifying the number of vertices in **cge_model.py**. Then, run the program simply with the `python cge-model.py` command.

## Output

During runtime, the program will display the scores of the best constructions in the current iteration.



