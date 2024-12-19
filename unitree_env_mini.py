import jax
from brax.envs import PipelineEnv, State
from jax import numpy as jnp
import brax
from brax.io import mjcf
from brax.base import Base, Motion, Transform
from brax import math
import numpy as np
import mujoco
from mujoco import mjx

class UnitreeEnvMini(PipelineEnv):
    def __init__(self):
        model = mujoco.MjModel.from_xml_path("unitree_g1/scene.xml")

        model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        model.opt.iterations = 6
        model.opt.ls_iterations = 6

        system = mjcf.load_model(model)

        n_frames = 10

        super().__init__(sys = system,
            backend='positional',
            n_frames = n_frames
        )

        self.initial_state = jnp.array(system.mj_model.keyframe('stand').qpos)
        self.nv = system.nv
        self.nu = system.nu
        self.control_range = system.actuator_ctrlrange

    def _get_obs(
            self, data: mjx.Data, prev_action: jnp.ndarray
    ) -> jnp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        position = data.qpos

        # external_contact_forces are excluded
        return jnp.concatenate([
            position,
            data.qvel,
            data.cinert[1:].ravel(),
            data.cvel[1:].ravel(),
            data.qfrc_actuator,
            prev_action
        ])

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        pipeline_state = self.pipeline_init(self.initial_state, jnp.zeros(self.nv))

        state_info = {
            "rng": rng,
        }
        metrics = {'distance': 0.0,
                   'reward': 0.0}

        obs = self._get_obs(pipeline_state, state_info["prev_torque"])
        reward, done, zero = jnp.zeros(3)
        state = State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=state_info
        )
        return state

    def step(self, state: State, action: jnp.ndarray):

        bottom_limit = self.control_range[:, 0]
        top_limit = self.control_range[:, 1]
        scaled_action = ( (action + 1) * (top_limit - bottom_limit) / 2 + bottom_limit )

        data0 = state.pipeline_state
        data = self.pipeline_step(data0, scaled_action)

        com_before = data0.subtree_com[1]
        com_after = data.subtree_com[1]
        velocity = (com_after - com_before) / self.dt
        forward_reward = 1.25 * velocity[0]

        min_z, max_z = (0.4, 0.8)
        is_healthy = jnp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(data.q[2] > max_z, 0.0, is_healthy)
        healthy_reward = 5.0 * is_healthy

        ctrl_cost = 0.05 * jnp.sum(jnp.square(action))

        obs = self._get_obs(data, action)
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy
        state.metrics.update(
            reward=reward,
            distance=jnp.linalg.norm(com_after),
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )