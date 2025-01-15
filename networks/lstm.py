#Need to make flax network in format that brax train is looking for.
#Need to make make_network_factory function that takes:
#observation_size: int,
#action_size: int,
#preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
#value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
#activation: networks.ActivationFn = linen.swish,
#policy_obs_key: str = 'state',
#value_obs_key: str = 'state',
#returns a PPONetworks with a policy, value and parametric action distribution


from typing import Sequence, Callable, Any

from brax.training import distribution
from brax.training import networks
from brax.training import types
import flax
import dataclasses
from flax import linen
import jax.numpy as jnp
import jax

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

@dataclasses.dataclass
class LSTMNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


def make_policy_network(
        param_size: int,
        obs_size: types.ObservationSize,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        activation: ActivationFn = linen.relu,
        kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
        layer_norm: bool = False,
        obs_key: str = 'state',
):
    policy_module = LSTM()

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    obs_size = networks._get_obs_state_size(obs_size, obs_key)
    dummy_obs = jnp.zeros((1, obs_size))
    return LSTMNetwork(init = lambda key: policy_module.init(key, dummy_obs), apply = apply)


@flax.struct.dataclass
class LSTMPPONetworks:
    policy_network: LSTMNetwork
    value_network: networks.FeedForwardNetwork

def make_inference_fn(networks: LSTMPPONetworks):
    def make_policy(
            params: types.Params, deterministic: bool = False
    ):
        policy_network = networks.policy_network
        parametric_action_distribution = networks.parametric_action_distribution
        def policy(observations, key_sample):
            param_subset = (params[0], params[1])
            logits = policy_network.apply( *param_subset, observations)
            if deterministic:
                return parametric_action_distribution.mode(logits), {}
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )
            return postprocessed_actions, {
                'log_prob': log_prob,
                'raw_action': raw_actions,
            }
        return policy
    return make_policy


def make_ppo_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
) -> LSTMPPONetworks:
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
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

    #policy hidden is hardcoded
    return LSTMPPONetworks(policy_network = policy_network,
                           value_network = value_network,
                           paramatric_action_distribution = parametric_action_distribution)
