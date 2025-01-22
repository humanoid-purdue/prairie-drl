
import dill
import jax
from brax import envs
from brax.io import html, mjcf, model
from nemo_grfless import NemoEnv

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
    with open(full_path, 'rb') as f:
        make_inference_fn = dill.load(f)

    inference_fn = make_inference_fn(saved_params)
    jit_inference_fn = jax.jit(inference_fn)

    n_steps = 1000
    ss=[]
    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        ss.append(state)
        rollout.append(state.pipeline_state)

    return env, rollout