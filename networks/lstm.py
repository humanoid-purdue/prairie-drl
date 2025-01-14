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


from typing import Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
import flax
from flax import linen

class LSTMNetwork:


@flax.struct.dataclass
class LSTMPPONetworks:
    policy_network: LSTMNetwork
    value_network: networks.FeedForwardNetwork

def make_inference_fn(networks: LSTMPPONetworks):
    def make_policy(
            params: types.Params, deterministics: bool = False
    ):
        policy_network = networks.policy_network
        def policy(observations, key_sample):
            logits = policy_network.apply(observations)




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
