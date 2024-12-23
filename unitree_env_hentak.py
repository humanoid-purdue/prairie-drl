import jax
from brax.envs import PipelineEnv, State
from jax import numpy as jnp
from brax.io import mjcf
import mujoco
from mujoco import mjx
import rewards
import numpy as np
from brax import math


DS_TIME = 0.2
SS_TIME = 0.5
BU_TIME = 0.05

class UnitreeEnvMini(PipelineEnv):
    def __init__(self):
        model = mujoco.MjModel.from_xml_path("unitree_g1/scene.xml")

        model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        model.opt.iterations = 6
        model.opt.ls_iterations = 6

        self.model = model

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
            self, data: mjx.Data, prev_action: jnp.ndarray, t = 0
    ) -> jnp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        position = data.qpos
        l_coeff, r_coeff = rewards.dualCycleCC(DS_TIME, SS_TIME, BU_TIME, t)
        # external_contact_forces are excluded
        return jnp.concatenate([
            position,
            data.qvel,
            data.cinert[1:].ravel(),
            data.cvel[1:].ravel(),
            data.qfrc_actuator,
            prev_action, l_coeff, r_coeff
        ])

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        pipeline_state = self.pipeline_init(self.initial_state, jnp.zeros(self.nv))

        state_info = {
            "rng": rng,
            "time": jnp.zeros(1),
            "l_force": jnp.zeros([1000, 6]),
            "r_force": jnp.zeros([1000, 6]),
            "l_orien": jnp.zeros([1000, 3]),
            "r_orien": jnp.zeros([1000, 3])
        }
        metrics = {'distance': 0.0,
                   'reward': 0.0}

        obs = self._get_obs(pipeline_state, jnp.zeros(self.nu), t = 0)
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

        #forward_reward = self.simple_vel_reward(data0, data) * 3.0
        period_reward, l_grf, r_grf = self.periodic_reward(state.info, data, data0)
        period_reward = period_reward[0] * 0.3

        upright_reward = self.upright_reward(data) * 1.0

        jl_reward = self.joint_limit_reward(data) * 5.0

        flatfoot_reward, l_vec, r_vec = self.flatfootReward(data)
        flatfoot_reward = flatfoot_reward * 10

        min_z, max_z = (0.4, 0.8)
        is_healthy = jnp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(data.q[2] > max_z, 0.0, is_healthy)
        healthy_reward = 5.0 * is_healthy

        ctrl_cost = 0.05 * jnp.sum(jnp.square(action))

        obs = self._get_obs(data, action, state.info["time"])
        reward = period_reward + healthy_reward - ctrl_cost + upright_reward + jl_reward + flatfoot_reward
        done = 1.0 - is_healthy
        com_after = data.subtree_com[1]
        state.metrics.update(
            reward=reward,
            distance=jnp.linalg.norm(com_after),
        )
        state.info["time"] += self.dt

        l_force = jnp.concatenate([l_grf[None, :], state.info["l_force"][0:-1, :]], axis = 0)
        r_force = jnp.concatenate([r_grf[None, :], state.info["r_force"][0:-1, :]], axis= 0)
        state.info["l_force"] = l_force
        state.info["r_force"] = r_force

        l_orien = jnp.concatenate([l_vec[None, :], state.info["l_orien"][0:-1, :]], axis = 0)
        r_orien = jnp.concatenate([r_vec[None, :], state.info["r_orien"][0:-1, :]], axis = 0)
        state.info["l_orien"] = l_orien
        state.info["r_orien"] = r_orien
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
        limit = self.joint_limit * 0.90

        # calculate the joint angles has larger or smaller than the limit
        out_of_limit = -jnp.clip(data1.q[7:] - limit[1:, 0], max=0., min=None)
        out_of_limit += jnp.clip(data1.q[7:] - limit[1:, 1], max=None, min=0.)

        # calculate the reward
        reward = jnp.sum(out_of_limit)
        return reward * -1

    def periodic_reward(self, info, data1, data0):
        t = info["time"]

        l_coeff, r_coeff = rewards.dualCycleCC(DS_TIME, SS_TIME, BU_TIME, t)

        l_contact_coeff = 2 * l_coeff -1
        r_contact_coeff = 2 * r_coeff - 1

        l_vel_coeff = 1 - l_coeff
        r_vel_coeff = 1 - r_coeff

        l_grf, r_grf = self.determineGRF(data1)
        l_nf = jnp.linalg.norm(l_grf[0:3])
        r_nf = jnp.linalg.norm(r_grf[0:3])

        l_nf = jnp.clip(l_nf, -400, 400)
        r_nf = jnp.clip(r_nf, -400, 400)

        def getVel(d1, d2, id):
            bp1 = d1.x
            bp2 = d2.x
            return (bp2.pos[id] - bp1.pos[id]) / self.dt

        l_vel = getVel(data0, data1, self.left_foot_id)
        #l_vel = jnp.clip(jnp.linalg.norm(l_vel), 0, 0.4)
        r_vel = getVel(data0, data1, self.right_foot_id)
        #r_vel = jnp.clip(jnp.linalg.norm(r_vel), 0, 0.4)

        vel_reward = l_vel_coeff * l_vel + r_vel_coeff * r_vel
        grf_reward = l_contact_coeff * l_nf + r_contact_coeff * r_nf


        return vel_reward * 1 + grf_reward * 0.1, l_grf, r_grf

    def crudeGRF(self, data):
        lpos = data.x.pos[self.left_foot_id]
        rpos = data.x.pos[self.right_foot_id]
        l_grf = jnp.where(lpos[2] < 0.04, 120, 0)
        r_grf = jnp.where(rpos[2] < 0.04, 120, 0)
        return l_grf, r_grf

    def flatfootReward(self, data):
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec_l = math.rotate(vec_tar, data.x.rot[self.left_foot_id])
        vec_r = math.rotate(vec_tar, data.x.rot[self.right_foot_id])
        rew = vec_l[2] + vec_r[2]
        return rew, vec_l, vec_r


    def determineGRF(self, data):

        forces = rewards.get_contact_forces(self.model, data)
        lfoot_grf, rfoot_grf = rewards.get_feet_forces(self.model, data, forces)

        return lfoot_grf, rfoot_grf