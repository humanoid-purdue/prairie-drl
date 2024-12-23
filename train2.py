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
import dill
from unitree_env_fixed_walk import UnitreeEnvMini

envs.register_environment('g1', UnitreeEnvMini)
env = envs.get_environment('g1')
eval_env = envs.get_environment('g1')

make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128))

pre_model_path = 'walk_policy'
pre_model = model.load_params(pre_model_path)

train_fn = functools.partial(
      ppo.train, num_timesteps=70000000, num_evals=40, episode_length = 2000,
       normalize_observations=True, unroll_length=20, num_minibatches=64,
      num_updates_per_batch=4, discounting=0.97, learning_rate=2.0e-4,
      entropy_cost=1e-3, num_envs=2048, batch_size=1024,
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

with open("inference_fn", 'wb') as f:
    dill.dump(make_inference_fn, f)