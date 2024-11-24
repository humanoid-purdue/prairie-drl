import unitree_env
import time
import itertools
import matplotlib.pyplot as plt
from datetime import datetime
import functools
from IPython.display import HTML
import jax
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from etils import epath
from flax import struct
from matplotlib import pyplot as plt
from ml_collections import config_dict
import mujoco
import mujoco.viewer
from mujoco import mjx

from unitree_env_pos import UnitreeEnvPos

envs.register_environment('g1', UnitreeEnvPos)
env = envs.get_environment('g1')
eval_env = envs.get_environment('g1')

make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128))

train_fn = functools.partial(
      ppo.train, num_timesteps=200000000,num_evals=10, episode_length = 1000,
       normalize_observations=True, unroll_length=20, num_minibatches=32,
      num_updates_per_batch=4, discounting=0.99, learning_rate=3.0e-4,
      entropy_cost=1e-3, num_envs=1024, batch_size=512,
      network_factory=make_networks_factory)

x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]

def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    plt.xlim([0, train_fn.keywords['num_timesteps']])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.title(f'y={y_data[-1]:.3f}')
    plt.plot(x_data, y_data)
    plt.show()

make_inference_fn, params, _= train_fn(environment=env,
                                       progress_fn=progress,
                                       eval_env=eval_env)

model.save_params("walk_policy", params)