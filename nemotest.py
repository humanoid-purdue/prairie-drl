import jax
from brax import envs
from brax.io import html, mjcf, model
import mujoco
import jax.numpy as jnp
from lstm_envs import *

def makeRollout(lstm = False, walk_forward = True, robot = "nemo4"):
    if lstm:
        from nemo_lstm import NemoEnv
    else:
        from nemo_env_pd import NemoEnv
    if robot == "nemo4" and lstm:
        model_n = mujoco.MjModel.from_xml_path("nemo4/scene.xml")
        c_env = Nemo4Env
    elif robot == "g2" and lstm:
        model_n = mujoco.MjModel.from_xml_path("g2/scene.xml")
        c_env = G2Env
    elif robot == "nemo4b" and lstm:
        print("nemo4b")
        model_n = mujoco.MjModel.from_xml_path("nemo4b/scene.xml")
        with open("input_files/nemo4.toml", "rb") as f:
            model_info = tomllib.load(f)
        class GenBotEnv(NemoEnv):
            def __init__(self):
                super().__init__(model_info=model_info)

        c_env = GenBotEnv
    pelvis_b_id = mujoco.mj_name2id(model_n, mujoco.mjtObj.mjOBJ_SITE, 'pelvis_back')
    pelvis_f_id = mujoco.mj_name2id(model_n, mujoco.mjtObj.mjOBJ_SITE, 'pelvis_front')

    envs.register_environment('nemo', c_env)
    env = envs.create(env_name='nemo')
    #print(env.observation_size, env.action_size)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)


    state = jit_reset(jax.random.PRNGKey(0))
    rollout = [state.pipeline_state]

    model_path = 'walk_policy'

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

    n_steps = 1000
    ss=[]
    for i in range(n_steps):
        #print(state.info["time"], state.info["halt_cmd"], state.info["phase"])
        if walk_forward:
            state.info["angvel"] = jax.numpy.array([0.0])
            state.info["velocity"] = jax.numpy.array([0.2, 0.0])
            state.info["event_period"] = jax.numpy.array([500 * 0.035, 100 * 0.035])
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

if __name__ == "__main__":
    makeRollout(lstm = True, robot = "nemo4")