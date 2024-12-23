import dill
import jax
from brax import envs
from brax.io import html, mjcf, model

from unitree_env_hentak import UnitreeEnvMini


envs.register_environment('g1', UnitreeEnvMini)
env_name = 'g1'
env = envs.create(env_name='g1')

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)


state = jit_reset(jax.random.PRNGKey(0))
rollout = [state.pipeline_state]

model_path = 'walk_policy'
full_path = "inference_fn"

# load saved model
saved_params = model.load_params(model_path)
rng = jax.random.PRNGKey(0)

def makeIFN():
    from brax.training.agents.ppo import networks as ppo_networks
    import functools
    from brax.training.acme import running_statistics
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=(128, 128, 128, 128))
    normalize = running_statistics.normalize
    obs_size = env.observation_size
    ppo_network = network_factory(
          obs_size, env.action_size, preprocess_observations_fn=normalize
      )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    return make_inference_fn

make_inference_fn = makeIFN()

inference_fn = make_inference_fn(saved_params)
jit_inference_fn = jax.jit(inference_fn)

# grab a trajectory
n_steps = 600
ss=[]
for i in range(n_steps):
    # if i == 750:
    #     command = jp.array([0.6,0.0])
    #     state.info['Control commands'] = command
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    ss.append(state)
    rollout.append(state.pipeline_state)
