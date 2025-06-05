import jax
from brax import envs
from brax.io import model
import mujoco
import jax.numpy as jnp
import mediapy


OBS_SIZE = 334
ACT_SIZE = 24
DT = 0.01

def generate_rollout(lstm=True):
    # Import check
    if lstm:
        from nemo_lstm import NemoEnv
    else:
        from nemo_env_pd import NemoEnv

    # Loading xml models
    model_n = mujoco.MjModel.from_xml_path("nemo4b/scene.xml")
    pelvis_b_id = mujoco.mj_name2id(model_n, mujoco.mjtObj.mjOBJ_SITE, 'pelvis_back')
    pelvis_f_id = mujoco.mj_name2id(model_n, mujoco.mjtObj.mjOBJ_SITE, 'pelvis_front')

    envs.register_environment('nemo', NemoEnv)
    env = envs.create(env_name='nemo')

    # JIT compile core functions
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Initialize state
    state = jit_reset(jax.random.PRNGKey(0))
    frames = []  # Store frames for video

    # Load policy
    saved_params = model.load_params('policies/walk_policy_acc7')
    rng = jax.random.PRNGKey(0)

    # Setup inference function
    def makeIFN():
        from brax.training.agents.ppo import networks as ppo_networks
        from networks.lstm import make_ppo_networks
        import functools
        from brax.training.acme import running_statistics
        mpn = make_ppo_networks
        network_factory = functools.partial(
            mpn,
            policy_hidden_layer_sizes=(512, 256, 256, 128))
        # normalize = running_statistics.normalize
        normalize = lambda x, y: x
        obs_size = OBS_SIZE
        ppo_network = network_factory(
            obs_size, ACT_SIZE, preprocess_observations_fn=normalize
        )
        make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
        return make_inference_fn

    inference_fn = makeIFN()(saved_params)
    jit_inference_fn = jax.jit(inference_fn)

    # Run simulation
    n_steps = 20000 
    for i in range(n_steps):
        # Update state info
        state.info["velocity"] = jax.numpy.array([0.4, 0.0])

        # Calculate facing direction
        data = state.pipeline_state
        pp1 = data.site_xpos[pelvis_f_id]
        pp2 = data.site_xpos[pelvis_b_id]
        facing_vec = (pp1 - pp2)[0:2]
        facing_vec = facing_vec / jnp.linalg.norm(facing_vec)
        state.info["angvel"] = facing_vec[1] * -2

        # Get action and step environment
        act_rng, rng = jax.random.split(rng)
        action, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, action)

        # Store frame
        frames.append(env.render(state.pipeline_state))

    # Save video using mediapy
    mediapy.write_video('nemo_simulation.mp4', frames, fps=60)
    # Could also do show_video
    # mediapy.show_video(frames, camera='track'), fps=1.0 / env.dt / render_every)
    return frames

if __name__ == "__main__":
    frames = generate_rollout(lstm=True)
    print("Simulation complete! Video saved as 'nemo_simulation.mp4'")
