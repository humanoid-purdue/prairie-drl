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
STEP_HEIGHT = 0.1

def rotateVec2(vec2, angle):
    rot_mat = jnp.array([[jnp.cos(angle), -1 * jnp.sin(angle)],[jnp.sin(angle), jnp.cos(angle)]])
    new_xy = jnp.matmul(rot_mat, vec2)
    return new_xy

class UnitreeEnvMini(PipelineEnv):
    def __init__(self):
        model = mujoco.MjModel.from_xml_path("unitree_g1/scene_pd.xml")

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
        self.left_foot_s3 = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "left_foot_p3")

        self.right_foot_s1 = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "right_foot_p1")
        self.right_foot_s2 = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "right_foot_p2")
        self.right_foot_s3 = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "right_foot_p3")



    def _get_obs(
            self, data: mjx.Data, prev_action: jnp.ndarray, t = 0, centroid_vel = jnp.array([0.4, 0.0]), face_vec = jnp.array([1.0, 0.0]), ang_vel = 0.0
    ) -> jnp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        position = data.qpos
        global_pos = data.x.pos[self.pelvis_id + 1:, :]
        center = data.x.pos[self.pelvis_id, :]
        local_pos = global_pos - center[None, :]
        local_pos = local_pos.flatten()
        #sites

        lp1 = data.site_xpos[self.left_foot_s1] - center
        lp2 = data.site_xpos[self.left_foot_s2] - center
        lp3 = data.site_xpos[self.left_foot_s3] - center

        rp1 = data.site_xpos[self.right_foot_s1] - center
        rp2 = data.site_xpos[self.right_foot_s2] - center
        rp3 = data.site_xpos[self.right_foot_s3] - center

        head = data.site_xpos[self.head_id] - center
        pel_front = data.site_xpos[self.pelvis_f_id] - center
        pel_back = data.site_xpos[self.pelvis_b_id] - center

        facing_vec = pel_front - pel_back
        facing_vec = facing_vec / jnp.linalg.norm(facing_vec)
        facing_vec = facing_vec.flatten()

        local_sites = jnp.concatenate([lp1, lp2, lp3, rp1, rp2, rp3, head, pel_front], axis = 0)

        l_grf, r_grf = self.determineGRF(data)

        l_coeff, r_coeff = rewards.dualCycleCC(DS_TIME, SS_TIME, BU_TIME, t)

        # external_contact_forces are excluded
        return jnp.concatenate([
            position,
            data.qvel,
            data.cinert[1:].ravel(),
            data.cvel[1:].ravel(),
            local_pos,
            l_grf, r_grf,
            jnp.array([ang_vel]),
            local_sites,
            facing_vec,
            prev_action, l_coeff, r_coeff, centroid_vel, face_vec
        ])

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        pipeline_state = self.pipeline_init(self.initial_state, jnp.zeros(self.nv))
        r = random.uniform(key, [2])
        mag = ( r[0] + 1 ) * 0.3
        unit = jnp.array([1, r[1] - 0.5])
        unit = unit / jnp.linalg.norm(unit)
        vel = unit * mag
        angular_velocity = 0.0 #z Rads / s
        state_info = {
            "rng": rng,
            "time": jnp.zeros(1),
            "count": jnp.zeros(1),
            "pos_xy": jnp.zeros([100, 2]),
            "pelvis_angle": jnp.zeros([200, 2]),
            "centroid_velocity": vel,
            "angular_velocity": angular_velocity,
            "facing_vec": unit,
        }
        metrics = {
                   'reward': 0.0,
                   'flatfoot_reward': 0.0,
                   'periodic_reward': 0.0,
                    'upright_reward': 0.0,
                    'limit_reward': 0.0,
                    'foot_orien_reward': 0.0,
                    'stride_reward': 0.0,
                    'pelvis_orien_reward': 0.0,
                    'velocity_reward': 0.0,
                    'swing_height_reward': 0.0,
                    'center_reward': 0.0,
                    'healthy_reward': 0.0,
                    'ctrl_reward': 0.0}

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

    def tanh2Action(self, action: jnp.ndarray):
        pos_t = action[:self.nu//2]
        vel_t = action[self.nu//2:]

        bottom_limit = self.joint_limit[1:, 0]
        top_limit = self.joint_limit[1:, 1]
        vel_sp = vel_t * 10
        pos_sp = ((pos_t + 1) * (top_limit - bottom_limit) / 2 + bottom_limit)

        act2 = jnp.concatenate([pos_sp, vel_sp])
        return act2

    def step(self, state: State, action: jnp.ndarray):
        scaled_action = self.tanh2Action(action)
        data0 = state.pipeline_state
        data1 = self.pipeline_step(data0, scaled_action)

        reward, done = self.rewards(state, data1, action)

        facing_vec = self.pelvisAngle(data1)
        pos_xy = data1.subtree_com[1][0:2]
        new_pxy = jnp.concatenate([pos_xy[None, :], state.info["pos_xy"][:-1, :]])
        state.info["pos_xy"] = new_pxy

        new_fv = jnp.concatenate([facing_vec[None, :], state.info["pelvis_angle"][:-1, :]])
        state.info["pelvis_angle"] = new_fv

        state.info["time"] += self.dt
        state.info["count"] += 1

        angular_displacement = state.info["angular_velocity"] * self.dt
        new_vel_vec = rotateVec2(state.info["centroid_velocity"], angular_displacement)
        new_unit_vec = new_vel_vec / jnp.linalg.norm(new_vel_vec)
        state.info["centroid_velocity"] = new_vel_vec
        state.info["facing_vec"] = new_unit_vec

        obs = self._get_obs(data1, action, state.info["time"], state.info["centroid_velocity"], state.info["facing_vec"], state.info["angular_velocity"])
        return state.replace(
            pipeline_state = data1, obs=obs, reward=reward, done=done
        )

    def rewards(self, state: State, data, action):

        data0 = state.pipeline_state

        reward_dict = {}

        period_reward = self.periodicReward(state.info, data, data0)
        period_reward = period_reward[0] * 0.6
        reward_dict["periodic_reward"] = period_reward

        upright_reward = self.upright_reward(data) * 10.0
        reward_dict["upright_reward"] = upright_reward

        jl_reward = self.joint_limit_reward(data) * 10.0
        reward_dict["limit_reward"] = jl_reward

        flatfoot_reward = self.flatfootReward(data)
        flatfoot_reward = flatfoot_reward * 1.0
        reward_dict["flatfoot_reward"] = flatfoot_reward

        footstep_reward = self.footstepOrienReward(state.info, data)[0] * 0.0
        reward_dict["foot_orien_reward"] = footstep_reward

        stride_length_reward = self.strideLengthReward(state.info, data)[0] * 200
        reward_dict["stride_reward"] = stride_length_reward

        facing_vec = self.pelvisAngle(data)

        pelvis_a_reward = self.pelvisAngleReward(facing_vec, state, state.info["facing_vec"]) * 6.0
        reward_dict["pelvis_orien_reward"] = pelvis_a_reward

        velocity_reward = self.velocityReward(state.info, data) * 10
        reward_dict["velocity_reward"] = velocity_reward

        swing_height_reward = self.swingHeightReward(state.info, data)[0] * 50
        reward_dict["swing_height_reward"] = swing_height_reward

        center_reward = self.centerReward(data) * 4
        reward_dict["center_reward"] = center_reward

        min_z, max_z = (0.4, 0.8)
        is_healthy = jnp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(data.q[2] > max_z, 0.0, is_healthy)
        healthy_reward = 5.0 * is_healthy
        reward_dict["healthy_reward"] = healthy_reward

        ctrl_reward = -0.00 * jnp.sum(jnp.square(action))
        reward_dict["ctrl_reward"] = ctrl_reward

        reward = 0.0
        for key in reward_dict.keys():
            reward += reward_dict[key]

        metric_dict = reward_dict.copy()
        metric_dict["reward"] = reward


        done = 1.0 - is_healthy
        state.metrics.update(
            **metric_dict
        )

        return reward, done

    def velocityReward(self, info, data):
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
        center = jnp.mean(self.joint_limit[1:, :], axis = 1)
        d_top = self.joint_limit[1:, 1] - center
        d_bottom = self.joint_limit[1:, 0] - center
        top = center + d_top * 0.8
        bottom = center + d_bottom * 0.8

        # calculate the joint angles has larger or smaller than the limit
        top_rew = jnp.clip(data1.q[7:] - top, min = 0, max = None)
        bottom_rew = jnp.clip(bottom - data1.q[7:], min = 0, max = None)

        # calculate the reward
        reward = jnp.sum(top_rew + bottom_rew)
        return reward * -1

    def swingHeightReward(self, info, data):
        t = info["time"]
        l_t, r_t = rewards.heightLimit(DS_TIME, SS_TIME, BU_TIME, STEP_HEIGHT, t)
        l_coeff, r_coeff = rewards.dualCycleCC(DS_TIME, SS_TIME, BU_TIME, t)

        lp1 = data.site_xpos[self.left_foot_s1]
        lp2 = data.site_xpos[self.left_foot_s2]
        lp3 = data.site_xpos[self.left_foot_s3]

        rp1 = data.site_xpos[self.right_foot_s1]
        rp2 = data.site_xpos[self.right_foot_s2]
        rp3 = data.site_xpos[self.right_foot_s3]

        l_h = ( lp1[2] + lp2[2] + lp3[2] ) / 3
        r_h = ( rp1[2] + rp2[2] + rp3[2]) / 3

        l_rew = jnp.clip(l_h - l_t, min = -10, max = 0)
        r_rew = jnp.clip(r_h - r_t, min = -10, max = 0)

        rew = l_rew * (1 - l_coeff) + r_rew * (1 - r_coeff)
        return rew

    def periodicReward(self, info, data1, data0):
        t = info["time"]

        l_coeff, r_coeff = rewards.dualCycleCC(DS_TIME, SS_TIME, BU_TIME, t)

        l_contact_coeff = 2 * l_coeff -1
        r_contact_coeff = 2 * r_coeff - 1

        gnd_vel_coeff = -5
        swing_vel_coeff = 0
        l_vel_coeff = swing_vel_coeff - l_coeff * (swing_vel_coeff - gnd_vel_coeff)
        r_vel_coeff = swing_vel_coeff - r_coeff * (swing_vel_coeff - gnd_vel_coeff)

        l_shuffle_coeff = l_coeff * -1
        r_shuffle_coeff = r_coeff * -1

        l_grf, r_grf = self.determineGRF(data1)
        l_nf = jnp.linalg.norm(l_grf[0:3])
        r_nf = jnp.linalg.norm(r_grf[0:3])


        l_nf = jnp.clip(l_nf, -400, 400)
        r_nf = jnp.clip(r_nf, -400, 400)

        def getVel(d1, d2, id):
            bp1 = d1.site_xpos
            bp2 = d2.site_xpos
            return (bp2[id] - bp1[id]) / self.dt

        l1_vel = getVel(data0, data1, self.left_foot_s1)
        l2_vel = getVel(data0, data1, self.left_foot_s2)

        r1_vel = getVel(data0, data1, self.right_foot_s1)
        r2_vel = getVel(data0, data1, self.right_foot_s2)

        l_spd = ( jnp.linalg.norm(l1_vel) + jnp.linalg.norm(l2_vel) ) / 2
        r_spd = ( jnp.linalg.norm(r1_vel) + jnp.linalg.norm(r2_vel) ) / 2

        l_shuffle = jnp.exp(jnp.linalg.norm(l1_vel - l2_vel) * -1 / 0.05)
        r_shuffle = jnp.exp(jnp.linalg.norm(r1_vel - r2_vel) * -1 / 0.05)


        vel_reward = l_vel_coeff * l_spd + r_vel_coeff * r_spd
        shuffle_reward = l_shuffle_coeff * l_shuffle + r_shuffle_coeff * r_shuffle
        grf_reward = l_contact_coeff * l_nf + r_contact_coeff * r_nf


        return vel_reward * 2 + grf_reward * 0.05 + shuffle_reward * 20

    def flatfootReward(self, data):
        vec_tar = jnp.array([0.0, 0.0, 1.0])

        def sites2Rew(p1, p2, p3):
            v1 = p1 - p3
            v2 = p2 - p3
            dot = jnp.cross(v1, v2)
            normal_vec = dot / jnp.linalg.norm(dot)
            return jnp.abs(normal_vec[2])

        lp1 = data.site_xpos[self.left_foot_s1]
        lp2 = data.site_xpos[self.left_foot_s2]
        lp3 = data.site_xpos[self.left_foot_s3]

        rp1 = data.site_xpos[self.right_foot_s1]
        rp2 = data.site_xpos[self.right_foot_s2]
        rp3 = data.site_xpos[self.right_foot_s3]

        rew = sites2Rew(lp1, lp2, lp3) + sites2Rew(rp1, rp2, rp3)

        return rew

    def strideLengthReward(self, info, data):
        t = info["time"]
        l_coeff, r_coeff = rewards.dualCycleCC(DS_TIME, SS_TIME, BU_TIME, t)
        #only check distance when both are ds
        ds_state = l_coeff * r_coeff
        vel_mag = jnp.linalg.norm(info["centroid_velocity"])
        stride_target = vel_mag * (DS_TIME + SS_TIME) * 1.0
        lp1 = data.site_xpos[self.left_foot_s1]
        lp2 = data.site_xpos[self.left_foot_s2]
        lp = (lp1 + lp2) / 2

        rp1 = data.site_xpos[self.right_foot_s1]
        rp2 = data.site_xpos[self.right_foot_s2]
        rp = (rp1 + rp2) / 2

        stride_length = jnp.linalg.norm(lp[0:2] - rp[0:2])
        close_reward = jnp.clip(stride_length, min=0, max=0.2) - 0.2

        reward = stride_target - stride_length
        reward = jnp.where(reward > 0, 0, reward)
        reward = reward * ds_state + close_reward
        return reward

    def determineGRF(self, data):

        forces = rewards.get_contact_forces(self.model, data)
        lfoot_grf, rfoot_grf = rewards.get_feet_forces(self.model, data, forces)

        return lfoot_grf, rfoot_grf

    def footstepOrienReward(self, info, data):
        t = info["time"]
        l_coeff, r_coeff = rewards.dualCycleCC(DS_TIME, SS_TIME, BU_TIME, t)
        def pos2Rew(p1, p2, target_orien):
            foot_orien = (p1 - p2)
            foot_xy = foot_orien[0:2] / jnp.linalg.norm(foot_orien[0:2])
            orien_rew = jnp.abs(jnp.sum(foot_xy * target_orien))
            return orien_rew

        lf1 = data.site_xpos[self.left_foot_s1].flatten()
        lf2 = data.site_xpos[self.left_foot_s2].flatten()

        rf1 = data.site_xpos[self.right_foot_s1].flatten()
        rf2 = data.site_xpos[self.right_foot_s2].flatten()

        l_rew = pos2Rew(lf1, lf2, info["facing_vec"])
        r_rew = pos2Rew(rf1, rf2, info["facing_vec"])

        t_c = 1
        b_c = 0.3
        l_coeff = l_coeff * (t_c - b_c) + b_c
        r_coeff = r_coeff * (t_c - b_c) + b_c

        rew = l_rew * l_coeff + r_rew * r_coeff
        return rew

    def centerReward(self, data):
        lp1 = data.site_xpos[self.left_foot_s1]
        lp2 = data.site_xpos[self.left_foot_s2]
        lp3 = data.site_xpos[self.left_foot_s3]

        rp1 = data.site_xpos[self.right_foot_s1]
        rp2 = data.site_xpos[self.right_foot_s2]
        rp3 = data.site_xpos[self.right_foot_s3]

        l_h = ( lp1[2] + lp2[2] + lp3[2] ) / 3
        r_h = ( rp1[2] + rp2[2] + rp3[2]) / 3

        pelvis_loc = data.x.pos[self.pelvis_id]

        l_norm = jnp.linalg.norm(pelvis_loc - l_h)
        r_norm = jnp.linalg.norm(pelvis_loc - r_h)

        rew = jnp.exp(-1 * jnp.abs(l_norm - r_norm) / 0.1)
        return rew