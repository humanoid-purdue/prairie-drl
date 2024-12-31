#from example from brax/training/agents/ppo/networks
#implement a make_network function which returns
from typing import Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
from typing import Any, Callable, Mapping, Sequence, Tuple
import dataclasses

def make_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
):
    return

@dataclasses.dataclass
class FeedForwardNetwork:
  init: Callable[..., Any]
  apply: Callable[..., Any]

@flax.struct.dataclass
class DualNetwork:
    policy_network = FeedForwardNetwork
