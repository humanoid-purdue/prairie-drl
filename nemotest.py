
import dill
import jax
from brax import envs
from brax.io import html, mjcf, model
from nemo_env_pd import NemoEnv

def makeRollout():
    envs.register_environment('nemo', NemoEnv)
    env_name = 'nemo'
    env = envs.create(env_name='nemo')

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)


    state = jit_reset(jax.random.PRNGKey(0))
    rollout = [state.pipeline_state]

    model_path = 'walk_policy'
    full_path = "inference_fn"

    # load saved model
    saved_params = model.load_params(model_path)
    rng = jax.random.PRNGKey(0)
    # Load saved inference function
    def makeIFN():
        from brax.training.agents.ppo import networks as ppo_networks
        import functools
        from brax.training.acme import running_statistics
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=(512, 256, 256, 128))
        # normalize = running_statistics.normalize
        normalize = lambda x, y: x
        obs_size = env.observation_size
        ppo_network = network_factory(
            obs_size, env.action_size, preprocess_observations_fn=normalize
        )
        make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
        return make_inference_fn

    make_inference_fn = makeIFN()

    inference_fn = make_inference_fn(saved_params)
    jit_inference_fn = jax.jit(inference_fn)

    n_steps = 4000
    ss=[]
    for i in range(n_steps):
        state.info["angvel"] = jax.numpy.array([0.0])
        state.info["velocity"] = jax.numpy.array([0.3, 0.0])
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        ss.append(state)
        rollout.append(state.pipeline_state)

    return env, rollout