import jax
from brax.envs import PipelineEnv, State
from jax import numpy as jnp
from brax.io import mjcf
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
            backend='mjx',
            n_frames = n_frames
        )

        self.initial_state = jnp.array(system.mj_model.keyframe('stand').qpos)
        self.nv = system.nv
        self.nu = system.nu
        self.control_range = system.actuator_ctrlrange
        self.joint_limit = jnp.array(model.jnt_range)

        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        self.head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'head_link')
        self.left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_ankle_roll_link')
        self.right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_ankle_roll_link')

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

        obs = self._get_obs(pipeline_state, jnp.zeros(self.nu))
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

        #constrained reward structure for forward walk:
        #Upright reward, vel target reward, joint limit reward

        #development plan:
        #1) Controlled velocity Failure
        #2) periodic reward
        #3) forced straight line footstep
        #4) randomized footsteps
        #5)
        #Parallel:
        #PD control
        #Remove qfrc_actuator

        bottom_limit = self.control_range[:, 0]
        top_limit = self.control_range[:, 1]
        scaled_action = ( (action + 1) * (top_limit - bottom_limit) / 2 + bottom_limit )

        data0 = state.pipeline_state
        data = self.pipeline_step(data0, scaled_action)


        forward_reward = self.simple_vel_reward(data0, data) * 3.0

        upright_reward = self.upright_reward(data) * 1.0

        jl_reward = self.joint_limit_reward(data) * 1.0

        min_z, max_z = (0.4, 0.8)
        is_healthy = jnp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(data.q[2] > max_z, 0.0, is_healthy)
        healthy_reward = 5.0 * is_healthy

        ctrl_cost = 0.05 * jnp.sum(jnp.square(action))

        obs = self._get_obs(data, action)
        reward = forward_reward + healthy_reward - ctrl_cost + upright_reward + jl_reward
        done = 1.0 - is_healthy
        com_after = data.subtree_com[1]
        state.metrics.update(
            reward=reward,
            distance=jnp.linalg.norm(com_after),
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )

    def velocity_reward(self, data0, data1):
        com_before = data0.subtree_com[1]
        com_after = data1.subtree_com[1]
        velocity = (com_after - com_before) / self.dt
        vel_target = jnp.array([0., 0.5])
        vel_err = (velocity[0:2] - vel_target) ** 2
        vel_err = vel_err * jnp.array([1, 2])
        return jnp.exp(jnp.sum(vel_err) * -10)

    def simple_vel_reward(self, data0, data1):
        com_before = data0.subtree_com[1]
        com_after = data1.subtree_com[1]
        velocity = (com_after - com_before) / self.dt
        vel_1 = jnp.where(velocity[0] > 0.5, 0.5, velocity[0])
        return vel_1

    def upright_reward(self, data1):
        body_pos = data1.x
        pelvis_xy = body_pos.pos[self.pelvis_id][0:2]
        head_xy = body_pos.pos[self.head_id][0:2]
        xy_err = jnp.linalg.norm(pelvis_xy - head_xy)
        return jnp.exp(xy_err * -10)

    def joint_limit_reward(self, data1):
        #within soft limit
        limit = self.joint_limit * 0.95

        # calculate the joint angles has larger or smaller than the limit
        out_of_limit = -jnp.clip(data1.q[7:] - limit[1:, 0], max=0., min=None)
        out_of_limit += jnp.clip(data1.q[7:] - limit[1:, 1], max=None, min=0.)

        # calculate the reward
        reward = jnp.sum(out_of_limit)
        return reward * -1