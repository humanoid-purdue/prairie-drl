import jax
from brax.envs import PipelineEnv, State
from jax import numpy as jnp
from brax.io import mjcf
import mujoco
from mujoco import mjx

import rewards
import numpy as np
from brax import math
from jax import random


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

        self.fsp = rewards.FootstepPlan(DS_TIME, SS_TIME, BU_TIME)


        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        self.pelvis_b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'pelvis_back')
        self.pelvis_f_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'pelvis_front')
        self.head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, 'head')

        self.left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_ankle_roll_link')
        self.right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_ankle_roll_link')

        self.left_foot_s1 = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "left_foot_p1")
        self.left_foot_s2 = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "left_foot_p2")

        self.right_foot_s1 = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "right_foot_p1")
        self.right_foot_s2 = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "right_foot_p2")



    def _get_obs(
            self, data: mjx.Data, prev_action: jnp.ndarray, t = 0, centroid_vel = jnp.array([0.4, 0.0]), face_vec = jnp.array([1.0, 0.0])
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
            prev_action, l_coeff, r_coeff, centroid_vel, face_vec
        ])

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        pipeline_state = self.pipeline_init(self.initial_state, jnp.zeros(self.nv))
        r = random.uniform(key, [2])
        mag = ( r[0] + 1 ) * 0.25
        unit = jnp.array([1, r[1] - 0.5])
        unit = unit / jnp.linalg.norm(unit)
        vel = unit * mag
        state_info = {
            "rng": rng,
            "time": jnp.zeros(1),
            "l_vec": jnp.zeros([4]),
            "fs_rew": jnp.zeros(1),
            "pos_xy": jnp.zeros([100, 2]),
            "pelvis_angle": jnp.zeros([100, 2]),
            "centroid_velocity": vel,
            "facing_vec": unit
        }
        metrics = {'distance': 0.0,
                   'reward': 0.0,
                   'flatfoot_reward': 0.0,
                   'periodic_reward': 0.0}

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

        upright_reward = self.upright_reward(data) * 5.0

        jl_reward = self.joint_limit_reward(data) * 5.0

        jm_reward = self.jointMagReward(data) * 0.0

        flatfoot_reward, l_vec, r_vec = self.flatfootReward(data)
        flatfoot_reward = flatfoot_reward * 5.0

        footstep_reward = self.footstepOrienReward(state.info, data)[0] * 0.2

        #simple_vel_reward, side_rew = self.simple_vel_reward(data0, data)
        #simple_vel_reward = simple_vel_reward * 2
        #side_rew = side_rew * 1

        facing_vec = self.pelvisAngle(data)

        pelvis_a_reward = self.pelvisAngleReward(facing_vec, state, state.info["facing_vec"]) * 3.0

        velocity_reward = self.velocity_reward(state.info, data) * 10

        min_z, max_z = (0.4, 0.8)
        is_healthy = jnp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(data.q[2] > max_z, 0.0, is_healthy)
        healthy_reward = 5.0 * is_healthy

        ctrl_cost = 0.05 * jnp.sum(jnp.square(action))

        obs = self._get_obs(data, action, state.info["time"], state.info["centroid_velocity"], state.info["facing_vec"])
        reward = period_reward + healthy_reward - ctrl_cost + jm_reward + footstep_reward + upright_reward + jl_reward + flatfoot_reward + velocity_reward + pelvis_a_reward
        done = 1.0 - is_healthy
        com_after = data.subtree_com[1]
        state.metrics.update(
            reward=reward,
            distance=jnp.linalg.norm(com_after),
        )


        lvec, rvec, l_coeff, r_coeff = self.fsp.getStepInfo(state.info["time"])
        state.info["time"] += self.dt
        state.info["l_vec"] = lvec
        state.info["fs_rew"] += footstep_reward

        pos_xy = data.subtree_com[1][0:2]
        new_pxy = jnp.concatenate([pos_xy[None, :], state.info["pos_xy"][:-1, :]])
        state.info["pos_xy"] = new_pxy

        new_fv = jnp.concatenate([facing_vec[None, :], state.info["pelvis_angle"][:-1, :]])
        state.info["pelvis_angle"] = new_fv

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )

    def velocity_reward(self, info, data):
        com = data.subtree_com[1]
        vel_target = info["centroid_velocity"]
        p0 = jnp.where(info["time"] < 1, jnp.array([0, 0]), info["pos_xy"][-1, :])
        t = jnp.where(info["time"] < 1, info["time"], 1)
        vel = ( com[0:2] - p0 ) / t
        vel_err = (vel - vel_target) ** 2
        return jnp.exp(jnp.sum(vel_err) * -10)


    def pelvisAngle(self, data):
        pelvis_c = data.site_xpos[self.pelvis_b_id][0:2]
        pelvis_f = data.site_xpos[self.pelvis_f_id][0:2]
        vec = pelvis_f - pelvis_c
        facing_vec = vec / jnp.linalg.norm(vec)
        return facing_vec

    def pelvisAngleReward(self, facing_vec, state, target):
        ave_angle = jnp.sum(state.info["pelvis_angle"], axis = 0)
        ave_angle = ave_angle / jnp.linalg.norm(ave_angle)
        vec = jnp.where(state.info["time"] < 1.0, facing_vec, ave_angle)
        vec = jnp.reshape(vec, [2])
        rew = jnp.sum(target * vec)
        return rew

    def upright_reward(self, data1):
        body_pos = data1.x
        pelvis_xy = body_pos.pos[self.pelvis_id][0:2]
        head_xy = data1.site_xpos[self.head_id][0:2]
        xy_err = jnp.linalg.norm(pelvis_xy - head_xy)
        return jnp.exp(xy_err * -30)

    def joint_limit_reward(self, data1):
        #within soft limit
        limit = self.joint_limit * 0.90

        # calculate the joint angles has larger or smaller than the limit
        out_of_limit = -jnp.clip(data1.q[7:] - limit[1:, 0], max=0., min=None)
        out_of_limit += jnp.clip(data1.q[7:] - limit[1:, 1], max=None, min=0.)

        # calculate the reward
        reward = jnp.sum(out_of_limit)
        return reward * -1

    def jointMagReward(self, data):
        prop = data.q[7:] / ( self.joint_limit[1:, 1] - self.joint_limit[1:, 0] )
        return jnp.sum(prop)

    def periodic_reward(self, info, data1, data0):
        t = info["time"]

        l_coeff, r_coeff = rewards.dualCycleCC(DS_TIME, SS_TIME, BU_TIME, t)

        l_contact_coeff = 2 * l_coeff -1
        r_contact_coeff = 2 * r_coeff - 1

        l_vel_coeff = 1 - l_coeff
        r_vel_coeff = 1 - r_coeff

        l_grf, r_grf = self.determineGRF(data1)
        l_nf1, r_nf1 = self.crudeGRF(data1)
        l_nf = jnp.linalg.norm(l_grf[0:3]) + l_nf1
        r_nf = jnp.linalg.norm(r_grf[0:3]) + r_nf1


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


        return vel_reward * 1 + grf_reward * 0.05, l_grf, r_grf

    def crudeGRF(self, data):
        lp1 = data.site_xpos[self.left_foot_s1]
        lp2 = data.site_xpos[self.left_foot_s2]

        rp1 = data.site_xpos[self.right_foot_s1]
        rp2 = data.site_xpos[self.right_foot_s2]

        l_grf = jnp.where(lp1[2] < 0.01, 1, 0) * jnp.where(lp2[2] < 0.01, 1, 0)
        r_grf = jnp.where(rp1[2] < 0.01, 1, 0) * jnp.where(rp2[2] < 0.01, 1, 0)
        return l_grf, r_grf

    def flatfootReward(self, data):
        def sites2Rew(p1, p2):
            delta = jnp.abs(p1[2] - p2[2])
            return jnp.exp( -1 * delta / 0.01)
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec_l = math.rotate(vec_tar, data.x.rot[self.left_foot_id])
        vec_r = math.rotate(vec_tar, data.x.rot[self.right_foot_id])

        lp1 = data.site_xpos[self.left_foot_s1]
        lp2 = data.site_xpos[self.left_foot_s2]

        rp1 = data.site_xpos[self.right_foot_s1]
        rp2 = data.site_xpos[self.right_foot_s2]



        rew = sites2Rew(lp1, lp2) + sites2Rew(rp1, rp2)

        return rew, vec_l, vec_r


    def determineGRF(self, data):

        forces = rewards.get_contact_forces(self.model, data)
        lfoot_grf, rfoot_grf = rewards.get_feet_forces(self.model, data, forces)

        l_filt, r_filt = self.crudeGRF(data)

        l_filt_grf = l_filt * lfoot_grf
        r_filt_grf = r_filt * rfoot_grf

        l_grf = jnp.where(jnp.linalg.norm(lfoot_grf) > 10, l_filt_grf, lfoot_grf)
        r_grf = jnp.where(jnp.linalg.norm(rfoot_grf) > 10, r_filt_grf, rfoot_grf)

        return lfoot_grf, rfoot_grf

    def footstepOrienReward(self, info, data):
        t = info["time"]
        l_coeff, r_coeff = rewards.dualCycleCC(DS_TIME, SS_TIME, BU_TIME, t)
        def pos2Rew(p1, p2, target_orien):
            foot_orien = (p1 - p2)
            foot_orien = foot_orien / jnp.linalg.norm(foot_orien)
            orien_rew = jnp.sum(foot_orien * target_orien)
            return orien_rew

        lf1 = data.site_xpos[self.left_foot_s1][0:2]
        lf2 = data.site_xpos[self.left_foot_s2][0:2]

        rf1 = data.site_xpos[self.right_foot_s1][0:2]
        rf2 = data.site_xpos[self.right_foot_s2][0:2]


        l_rew = pos2Rew(lf1, lf2, info["facing_vec"])
        r_rew = pos2Rew(rf1, rf2, info["facing_vec"])

        rew = l_rew * l_coeff + r_rew * r_coeff

        return rew