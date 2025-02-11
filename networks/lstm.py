from typing import Sequence, Tuple, Callable, Any

from brax.training import distribution
from brax.training import networks
from brax.training.agents.ppo.networks import PPONetworks
from brax.training.distribution import ParametricDistribution, TanhBijector, NormalDistribution
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
from flax import linen as nn
import jax
import jax.numpy as jnp
from jaxopt import *

#LSTM Network done by passing hidden and cell states as action
#Need custom flax nn, make_policy_network, distribution, and make_ppo_network
#Action size consits of 128 + 128 + a size

HIDDEN_SIZE = 256
DEPTH = 1

class StackedLSTM(nn.Module):
    param_size: int
    kernel_init: jax.nn.initializers.lecun_uniform()
    def setup(self):
        self.nn_in1 = nn.Dense(512, name = "i1", kernel_init=self.kernel_init)
        self.nn_in2 = nn.Dense(HIDDEN_SIZE, name = "i2", kernel_init=self.kernel_init)
        self.nn_mi = nn.Dense(HIDDEN_SIZE, name = "mid", kernel_init=self.kernel_init)
        self.nn_ed = nn.Dense(self.param_size, name = "end", kernel_init=self.kernel_init)
        self.lstms = [nn.OptimizedLSTMCell(HIDDEN_SIZE,
                        name = "lstm_{}".format(c)) for c in range(DEPTH * 2)]
        return

    def __call__(self, x):
        bs = x.shape[:-1]
        carry = x[..., :4 * HIDDEN_SIZE * DEPTH]
        obs = x[..., 4 * HIDDEN_SIZE * DEPTH:]
        y = nn.swish(self.nn_in1(obs))
        y = nn.swish(self.nn_in2(y))
        y0 = y.copy()
        hidden = carry[..., :2 * HIDDEN_SIZE * DEPTH]
        hidden = jnp.reshape(hidden, bs + (2 * DEPTH, HIDDEN_SIZE,))
        cell = carry[..., 2 * HIDDEN_SIZE * DEPTH:]
        cell = jnp.reshape(cell, bs + (2 * DEPTH, HIDDEN_SIZE,))

        hidden_next = jnp.zeros(bs + (2 * DEPTH, HIDDEN_SIZE,))
        cell_next = jnp.zeros(bs + (2 * DEPTH, HIDDEN_SIZE,))
        for i in range(DEPTH):
            state, y = self.lstms[i]((hidden[..., i, :], cell[..., i, :]), y)
            hidden_next = hidden_next.at[..., i, :].set(state[0])
            cell_next = cell_next.at[..., i, :].set(state[1])
        y1 = nn.swish(self.nn_mi(y + y0))
        y = y1.copy()
        for j in range(DEPTH, 2 * DEPTH):
            state, y = self.lstms[j]((hidden[..., j, :], cell[..., j, :]), y)
            hidden_next = hidden_next.at[..., j, :].set(state[0])
            cell_next = cell_next.at[..., j, :].set(state[1])
        y2 = self.nn_ed(y + y1)
        hidden_next = jnp.reshape(hidden_next, bs + (-1,))
        cell_next = jnp.reshape(cell_next, bs + (-1,))
        output = jnp.concat([hidden_next, cell_next, y2], axis = -1)
        return output

class LSTMTanhDistribution(ParametricDistribution):
    """Normal distribution followed by tanh."""
    def __init__(self, event_size, min_std=0.001, var_scale=1):
        super().__init__(
            param_size=2 * event_size,
            postprocessor=TanhBijector(),
            event_ndims=1,
            reparametrizable=True,
        )
        self._min_std = min_std
        self._var_scale = var_scale

    def create_dist(self, parameters):
        loc, scale = jnp.split(parameters[..., 4 * HIDDEN_SIZE * DEPTH:],
                               2, axis=-1)
        scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
        return NormalDistribution(loc=loc, scale=scale)

    def postprocess(self, event):
        #pass identity of the first 256 and only apply forward to remaining
        iden_event = event[..., :4 * HIDDEN_SIZE * DEPTH]
        action_event = self._postprocessor.forward(
              event[..., 4 * HIDDEN_SIZE * DEPTH:])
        y = jnp.concat([iden_event, action_event], axis = -1)
        return y

    def log_prob(self, parameters, actions):
        """Compute the log probability of actions."""
        dist = self.create_dist(parameters)
        log_probs = dist.log_prob(actions[..., 4 * HIDDEN_SIZE * DEPTH : ])
        log_probs -= self._postprocessor.forward_log_det_jacobian(
            actions[..., 4 * HIDDEN_SIZE * DEPTH : ])
        if self._event_ndims == 1:
            log_probs = jnp.sum(log_probs, axis=-1)  # sum over action dimension
        return log_probs

    def sample_no_postprocessing(self, parameters, seed):
        sample_act = self.create_dist(parameters).sample(seed=seed)
        carry = parameters[..., :4 * HIDDEN_SIZE * DEPTH]
        y = jnp.concat([carry, sample_act], axis = -1)
        return y


def make_policy_network(
    param_size: int,
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
    obs_key: str = 'state',
) -> networks.FeedForwardNetwork:
    """Creates a policy network."""
    policy_module = StackedLSTM(param_size = param_size, kernel_init = kernel_init)

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
        return policy_module.apply(policy_params, obs)

    obs_size = networks._get_obs_state_size(obs_size, obs_key)
    dummy_obs = jnp.zeros((1, obs_size))
    return networks.FeedForwardNetwork(
          init=lambda key: policy_module.init(key, dummy_obs), apply=apply
    )


def make_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 6,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
) -> PPONetworks:
    parametric_action_distribution = LSTMTanhDistribution(
        event_size=action_size
    )
    policy_network = make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
        obs_key=policy_obs_key,
    )
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        obs_key=value_obs_key,
    )
    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )