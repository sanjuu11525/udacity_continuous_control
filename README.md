
# Udacity Reinforcement Learning Continuous Control Project

## Introduction
This repository is to solve the Udacity exercise, Reacher, with different reinforcement learning DDPG models. For this project, an agent has been trained to keep a robot's arm in the ball-like target location.

### The Environment
The state space has 33 dimensions to describe position, rotation, velocity, and angular velocities of the arm. Continuous action space with 4 outputs with respect to torque to two joints.

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. So, the agent is to accumulate rewards to keep the arm in the target location. The task is solved if, the agent gets an average score of +30 over 100 consecutive episodes.

## Getting Start **One (1) Agent**

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system: 


    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)


Please make sure each ```file_path``` in ```train_dqn.ipynb``` and ```eval.py``` available.

## Installation

1. Clone the repository and initialize the submodules.

```
git clone https://github.com/sanjuu11525/udacity_continuous_control.git
cd udacity_continuous_control 
```

2. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ,JupytorLab and  create a new conda env.

```
conda create -n ENV_NAME python=3.6
conda activate ENV_NAME
pip install jupyterlab
```

3. Install the project requirements.

```
pip install -r requirements.txt
```
## Running the Code

1. This repository is for Udacity Continuous_Control project. Some implementation is publicly avaialble by Udacity. Please visit the reference for more information.

2. Train the agent with implemented code in ```train.ipynb```.

3. Evaluate trained models by runnung ```python eval.py```. Please turn your jupytor down when evaluating training result. Otherwise, resource conficts.

4. Pretrained models are in ```./checkpoint```.

5. DDPG is available and tested. A2C is still in prototyping.


## Reference

[1]https://github.com/udacity/deep-reinforcement-learning#dependencies