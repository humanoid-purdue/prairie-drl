#Need to make flax network in format that brax train is looking for.
#Need to make make_network_factory function that takes:
#observation_size: int,
#action_size: int,
#preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
#value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
#activation: networks.ActivationFn = linen.swish,
#policy_obs_key: str = 'state',
#value_obs_key: str = 'state',
#returns a PPONetworks with a policy, value and