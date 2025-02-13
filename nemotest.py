import jax
from brax import envs
from brax.io import html, mjcf, model
import mujoco
import jax.numpy as jnp

def makeRollout(lstm = False):
    if lstm:
        from nemo_lstm import NemoEnv
    else:
        from nemo_env_pd import NemoEnv
    model_n = mujoco.MjModel.from_xml_path("nemo2/scene.xml")
    pelvis_b_id = mujoco.mj_name2id(model_n, mujoco.mjtObj.mjOBJ_SITE, 'pelvis_back')
    pelvis_f_id = mujoco.mj_name2id(model_n, mujoco.mjtObj.mjOBJ_SITE, 'pelvis_front')

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
        from networks.lstm import make_ppo_networks
        import functools
        from brax.training.acme import running_statistics
        if lstm:
            mpn = make_ppo_networks
        else:
            mpn = ppo_networks.make_ppo_networks
        network_factory = functools.partial(
            mpn,
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
        state.info["velocity"] = jax.numpy.array([0.4, 0.0])
        data = state.pipeline_state
        pp1 = data.site_xpos[pelvis_f_id]
        pp2 = data.site_xpos[pelvis_b_id]
        facing_vec = (pp1 - pp2)[0:2]
        facing_vec = facing_vec / jnp.linalg.norm(facing_vec)
        state.info["angvel"] = jnp.array([facing_vec[1] * -2])
        #state.info["angvel"] = jnp.array([jnp.where(facing_vec[1] > 0, -0.4, 0.4)])

        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        ss.append(state)
        rollout.append(state.pipeline_state)

    return env, rollout