import dill
import time
import itertools
import mediapy as media
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
env_name = 'g1'
env = envs.get_environment(env_name)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

command = jp.array([1.0,0.0])

state = jit_reset(jax.random.PRNGKey(0))
state.info['Control commands'] = command
rollout = [state.pipeline_state]

model_path = 'walk_policy'
full_path = "inference_fn"

# load saved model
saved_params = model.load_params(model_path)
rng = jax.random.PRNGKey(0)
# Load saved inference function
with open(full_path, 'rb') as f:
    make_inference_fn = dill.load(f)

inference_fn = make_inference_fn(saved_params)
jit_inference_fn = jax.jit(inference_fn)

# grab a trajectory
n_steps = 600
render_every = 2
ss=[]
for i in range(n_steps):
    if i == 200:
        command = jp.array([1.0,0.3])
        state.info['Control commands'] = command
    if i == 400:
        command = jp.array([1.0,-0.3])
        state.info['Control commands'] = command
    # if i == 750:
    #     command = jp.array([0.6,0.0])
    #     state.info['Control commands'] = command
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    ss.append(state)
    rollout.append(state.pipeline_state)

media.show_video(env.render(rollout[::render_every], camera='track'), fps=1.0 / env.dt / render_every)
# %%