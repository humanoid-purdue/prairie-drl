
from typing import Sequence, Tuple, Callable, Any

from brax.training import distribution
from brax.training import networks
from brax.training.agents.ppo.networks import PPONetworks, make_inference_fn
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
from flax import linen as nn
import jax
import jax.numpy as jnp
from jaxopt import EqualityConstrainedQP


class OptNet(linen.Module): #No parameters, hardcoded first
    param_size: int
    qp_size: int
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    def setup(self):
        self.dense1 = nn.Dense(256, name = "hidden_1",
                               kernel_init=self.kernel_init, use_bias=True)
        self.dense2 = nn.Dense(128, name = "hidden_2",
                               kernel_init=self.kernel_init, use_bias=True)
        self.a_1 = nn.Dense(self.qp_size * self.qp_size)
        self.b_1 = nn.Dense(self.qp_size)
        self.q_mat_1 = self.param('qmat1',
                                  lambda rng: jnp.eye(self.qp_size))
        self.c_vec_1 = self.param('cvec1',
                                  lambda rng: jnp.zeros([self.qp_size]))
        self.qp1 = EqualityConstrainedQP(tol=1e-5, refine_regularization=3.,
                                         refine_maxiter=10)
        qpf1 = lambda a, b, q, c: (self.qp1.run(params_obj=(q, c),
                                               params_eq=(a, b)).params).primal
        self.b_qpf1 = jax.vmap(qpf1, ((0, 1), (0, 1), None, None), (0, 1))

        self.dense3 = nn.Dense(128, name = "hidden_3",
                               kernel_init=self.kernel_init, use_bias=True)
        self.dense4 = nn.Dense(128, name = "hidden_4",
                               kernel_init=self.kernel_init, use_bias=True)
        self.a_2 = nn.Dense(self.qp_size * self.qp_size)
        self.b_2 = nn.Dense(self.qp_size)
        self.q_mat_2 = self.param('qmat2',
                                  lambda rng: jnp.eye(self.qp_size))
        self.c_vec_2 = self.param('cvec2',
                                  lambda rng: jnp.zeros([self.qp_size]))
        self.qp2 = EqualityConstrainedQP(tol=1e-5, refine_regularization=3.,
                                         refine_maxiter=10)
        qpf2 = lambda a, b, q, c: (self.qp2.run(params_obj=(q, c),
                                               params_eq=(a, b)).params).primal
        self.b_qpf2 = jax.vmap(qpf2, ((0, 1), (0, 1), None, None), (0, 1))

        self.dense5 = nn.Dense(128, name = "hidden_5",
                               kernel_init=self.kernel_init, use_bias=True)
        self.dense6 = nn.Dense(self.param_size, name = "hidden_6",
                               kernel_init=self.kernel_init, use_bias=True)

    def __call__(self, x):
        y1 = nn.swish(self.dense1(x))
        y2 = nn.swish(self.dense2(y1))
        A1 = self.a_1(y2)
        A1 = jnp.reshape(A1, A1.shape[:-1] + (self.qp_size, self.qp_size))
        b1 = self.b_1(y2)
        qp_sol1 = self.b_qpf1(A1, b1, self.q_mat_1, self.c_vec_1)
        y3 = nn.swish(self.dense3(qp_sol1) + y2)
        y4 = nn.swish(self.dense4(y3))
        A2 = self.a_1(y4)
        A2 = jnp.reshape(A2, A1.shape[:-1] + (self.qp_size, self.qp_size))
        b2 = self.b_2(y4)
        qp_sol2 = self.b_qpf1(A2, b2, self.q_mat_2, self.c_vec_2)
        y5 = nn.swish(self.dense5(qp_sol2) + y4)
        y6 = self.dense6(y5)
        return y6


#make_inference_fn() and PPONetworks can be shared from default
#make ppo networks needs to be replaced. Initialization for constructors done below

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
  policy_module = OptNet(param_size = param_size, qp_size = 40, kernel_init = kernel_init)

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
    parametric_action_distribution = distribution.NormalTanhDistribution(
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