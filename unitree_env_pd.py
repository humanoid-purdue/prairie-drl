import jax
from brax.envs import PipelineEnv, State
from jax import numpy as jnp
from brax.io import mjcf
import mujoco
from mujoco import mjx

import rewards
from jax import random


DS_TIME = 0.2
SS_TIME = 0.5
BU_TIME = 0.05
STEP_HEIGHT = 0.10

def rotateVec2(vec2, angle):
    rot_mat = jnp.array([[jnp.cos(angle), -1 * jnp.sin(angle)],[jnp.sin(angle), jnp.cos(angle)]])
    new_xy = jnp.matmul(rot_mat, vec2)
    return new_xy

metrics_dict = {
                   'reward': 0.0,
                   'flatfoot_reward': 0.0,
                   'periodic_reward': 0.0,
                    'upright_reward': 0.0,
                    'limit_reward': 0.0,
                    'swing_height_reward': 0.0,
                    'healthy_reward': 0.0,
                    'stride_reward': 0.0,
                    'velocity_reward': 0.0,
                    'facing_reward': 0.0}

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
            self, data0, data1, prev_action: jnp.ndarray, state = None
    ) -> jnp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        def getCoords(data_):

            global_pos = data_.x.pos[self.pelvis_id + 1:, :]
            center = data_.x.pos[self.pelvis_id, :]
            local_pos = global_pos - center[None, :]
            local_pos = local_pos.flatten()
            #sites

            lp1 = data_.site_xpos[self.left_foot_s1] - center
            lp2 = data_.site_xpos[self.left_foot_s2] - center
            lp3 = data_.site_xpos[self.left_foot_s3] - center

            rp1 = data_.site_xpos[self.right_foot_s1] - center
            rp2 = data_.site_xpos[self.right_foot_s2] - center
            rp3 = data_.site_xpos[self.right_foot_s3] - center

            head = data_.site_xpos[self.head_id] - center
            pel_front = data_.site_xpos[self.pelvis_f_id] - center

            com_offset = (data_.subtree_com[1] - center).flatten()

            local_sites = jnp.concatenate([local_pos, lp1, lp2, lp3, rp1, rp2, rp3, head, pel_front, com_offset], axis = 0)
            return local_sites

        position = data1.qpos
        prev_sites = getCoords(data0)
        current_sites = getCoords(data1)
        center = data1.x.pos[self.pelvis_id, :]

        #l_grf, r_grf = self.determineGRF(data1)

        if state is not None:
            t = state.info["time"]
            fstep_plan = state.info["footstep_plan"]
            pointer = state.info["pointer"]
            step0 = jnp.sum(pointer[:, None] * fstep_plan, axis = 0)
            step1 = jnp.sum(jnp.roll(pointer, 1)[:, None] * fstep_plan, axis = 0)
            step2 = jnp.sum(jnp.roll(pointer, 2)[:, None] * fstep_plan, axis = 0)

            step0 = step0 - center[0:2]
            step1 = step1 - center[0:2]
            step2 = step2 - center[0:2]
            steps = jnp.concatenate([step0, step1, step2], axis = 0)
        else:
            t = 0
            steps = jnp.zeros([6])

        l_coeff, r_coeff = rewards.dualCycleCC(DS_TIME, SS_TIME, BU_TIME, t)

        # external_contact_forces are excluded
        angvel = data1.xd.ang[self.pelvis_id, :]

        com0 = data1.subtree_com[0]
        com1 = data1.subtree_com[1]
        vel = (com1 - com0) / self.dt

        return jnp.concatenate([
            position,
            data1.qvel,
            angvel,
            vel,
            prev_sites, current_sites,
            prev_action, l_coeff, r_coeff, steps
        ])

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        pipeline_state = self.pipeline_init(self.initial_state, jnp.zeros(self.nv))
        footstep_plan, pointer, weight, leg = rewards.sequentialFootstepPlan()

        state_info = {
            "rng": rng,
            "time": jnp.zeros(1),
            "count": jnp.zeros(1),
            "pos_xy": jnp.zeros([100, 2]),
            "footstep_plan": footstep_plan,
            "pointer": pointer,
            "prev_lin_mom": jnp.zeros(3),
            "prev_ang_mom": jnp.zeros(3),
            "hit_time": 0.0,
            "step_weight": weight,
            "l_xy": jnp.zeros(2),
            "r_xy": jnp.zeros(2),
            "fplan_reward": 0.0,
            "leg": leg,
            "centroid_velocity": jnp.array([0.4, 0])
        }
        metrics = metrics_dict.copy()

        obs = self._get_obs(pipeline_state, pipeline_state, jnp.zeros(self.nu))
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

        state.info["time"] += self.dt
        state.info["count"] += 1

        obs = self._get_obs(data0, data1, action, state = state)
        return state.replace(
            pipeline_state = data1, obs=obs, reward=reward, done=done
        )

    def rewards(self, state: State, data, action):

        data0 = state.pipeline_state

        reward_dict = {}

        period_reward = self.periodicReward(state.info, data, data0)
        period_reward = period_reward[0] * 0.6
        reward_dict["periodic_reward"] = period_reward

        upright_reward = self.upright_reward(data) * 8.0
        reward_dict["upright_reward"] = upright_reward

        jl_reward = self.joint_limit_reward(data) * 10.0
        reward_dict["limit_reward"] = jl_reward

        flatfoot_reward = self.flatfootReward(data)
        flatfoot_reward = flatfoot_reward * 1.5
        reward_dict["flatfoot_reward"] = flatfoot_reward

        swing_height_reward = self.swingHeightReward(state.info, data)[0] * 300
        reward_dict["swing_height_reward"] = swing_height_reward

        min_z, max_z = (0.4, 0.8)
        is_healthy = jnp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(data.q[2] > max_z, 0.0, is_healthy)
        healthy_reward = 5.0 * is_healthy
        reward_dict["healthy_reward"] = healthy_reward

        #footplan_reward = self.footplanReward(data, state) * 30
        #reward_dict["footplan_reward"] = footplan_reward

        facing_reward = self.facingReward(data, jnp.array([1., 0.])) * 4.0
        reward_dict["facing_reward"] = facing_reward

        stride_reward = self.strideLengthReward(state.info, data)[0] * 200
        reward_dict["stride_reward"] = stride_reward

        vel_reward = self.velocityReward(state.info, data) * 10.0
        reward_dict["velocity_reward"] = vel_reward

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
        pelvis_vec = pelvis_f - pelvis_c

        lf1 = data.site_xpos[self.left_foot_s1].flatten()[0:2]
        lf2 = data.site_xpos[self.left_foot_s2].flatten()[0:2]

        l_vec = lf1 - lf2

        rf1 = data.site_xpos[self.right_foot_s1].flatten()[0:2]
        rf2 = data.site_xpos[self.right_foot_s2].flatten()[0:2]

        r_vec = rf1 - rf2

        l_vec = l_vec / jnp.linalg.norm(l_vec)
        r_vec = r_vec / jnp.linalg.norm(r_vec)
        pelvis_vec = pelvis_vec / jnp.linalg.norm(pelvis_vec)

        ave_vec = l_vec + r_vec + pelvis_vec * 2
        ave_vec = ave_vec / jnp.linalg.norm(ave_vec)

        return ave_vec

    def facingReward(self, data, target):

        ave_vec = self.pelvisAngle(data)

        #angle = jnp.arccos(jnp.clip(jnp.sum(target * ave_vec), min = -1, max = 1))
        #rew = jnp.exp(-1 * angle / 1.5)
        rew = jnp.sum(target * ave_vec)
        rew = jnp.clip(rew, min = -1, max = 0.995)

        lf1 = data.site_xpos[self.left_foot_s1].flatten()[0:2]
        lf2 = data.site_xpos[self.left_foot_s2].flatten()[0:2]

        l_vec = lf1 - lf2

        rf1 = data.site_xpos[self.right_foot_s1].flatten()[0:2]
        rf2 = data.site_xpos[self.right_foot_s2].flatten()[0:2]

        r_vec = rf1 - rf2

        l_vec = l_vec / jnp.linalg.norm(l_vec)
        r_vec = r_vec / jnp.linalg.norm(r_vec)

        pelvis_c = data.site_xpos[self.pelvis_b_id][0:2]
        pelvis_f = data.site_xpos[self.pelvis_f_id][0:2]
        pelvis_vec = pelvis_f - pelvis_c

        pelvis_vec = pelvis_vec / jnp.linalg.norm(pelvis_vec)

        lr_delta =  jnp.sum(l_vec * r_vec)
        tol = 0.54
        cost = lr_delta - tol
        cost = jnp.clip(cost, min = -1, max = 0)

        l_dot = jnp.sum(pelvis_vec * l_vec)
        r_dot = jnp.sum(pelvis_vec * r_vec)
        # cost for being more than 60 degrees from any leg
        l_cost = jnp.clip(l_dot - 0.54, min = -1, max = 0)
        r_cost = jnp.clip(r_dot - 0.54, min = -1, max = 0)

        rew = rew + cost + l_cost + r_cost

        #angle = jnp.arccos(jnp.sum(target * ave_vec))
        #rew = jnp.exp(-1 * angle / 0.5)
        #rew = jnp.clip(rew, min = 0, max = 0.85)

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

        l_shuffle = jnp.exp(jnp.linalg.norm(l1_vel - l2_vel) * -1 / 0.01)
        r_shuffle = jnp.exp(jnp.linalg.norm(r1_vel - r2_vel) * -1 / 0.01)


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
        stride_target = jnp.clip(stride_target, min = 0.3, max = None)
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
        reward = reward * 0.2 * ds_state + close_reward
        return reward

    def determineGRF(self, data):

        forces = rewards.get_contact_forces(self.model, data)
        lfoot_grf, rfoot_grf = rewards.get_feet_forces(self.model, data, forces)

        return lfoot_grf, rfoot_grf

    def centerReward(self, data):
        lp1 = data.site_xpos[self.left_foot_s1]
        lp2 = data.site_xpos[self.left_foot_s2]
        lp3 = data.site_xpos[self.left_foot_s3]

        rp1 = data.site_xpos[self.right_foot_s1]
        rp2 = data.site_xpos[self.right_foot_s2]
        rp3 = data.site_xpos[self.right_foot_s3]

        l_h = ( lp1[2] + lp2[2] + lp3[2] ) / 3
        r_h = ( rp1[2] + rp2[2] + rp3[2]) / 3

        pelvis_loc = data.x.pos[self.pelvis_id-1]

        l_norm = jnp.linalg.norm(pelvis_loc - l_h)
        r_norm = jnp.linalg.norm(pelvis_loc - r_h)

        rew = jnp.exp(-1 * jnp.abs(l_norm - r_norm) / 0.1)
        return rew

    def angvelReward(self, data, state):
        #if the facing angle and target vector is more than 40 degrees, give capped reward for angvel
        pelvis_angvel = data.xd.ang[self.pelvis_id, :]
        z_angvel = pelvis_angvel[2]
        facing_vec = self.pelvisAngle(data)
        target_vec = state.info["facing_vec"]
        #delta = jnp.sum(state.info["facing_vec"] * facing_vec)
        angle = jnp.arctan2(facing_vec[0] * target_vec[1] - facing_vec[1] * target_vec[0],
                            facing_vec[0] * target_vec[0] + facing_vec[1] * target_vec[1])
        #Positive for ccw

        reward_ccw = jnp.clip(z_angvel, min = 0, max = 0.3)
        reward_cw = jnp.clip(z_angvel, min = -0.3, max = 0)
        dirac_delta = jnp.where(jnp.abs(angle) > 0.4, 1, 0)
        ccw = jnp.where(angle > 0, 1, 0)
        cw = jnp.where(angle < 0, 1, 0)
        reward = dirac_delta * (reward_ccw * ccw + reward_cw * cw)
        return reward

    def footplanReward(self, data, state):
        lp1 = data.site_xpos[self.left_foot_s1]
        lp2 = data.site_xpos[self.left_foot_s2]
        lp = (lp1 + lp2) / 2

        rp1 = data.site_xpos[self.right_foot_s1]
        rp2 = data.site_xpos[self.right_foot_s2]
        rp = (rp1 + rp2) / 2



        pp = data.x.pos[self.pelvis_id - 1][0:2]

        target = jnp.sum(state.info["pointer"][:, None] * state.info["footstep_plan"], axis = 0)

        l_dist = jnp.linalg.norm(target - lp[0:2])
        r_dist = jnp.linalg.norm(target - rp[0:2])

        leg = jnp.sum(state.info["leg"] * state.info["pointer"])

        min_dist = l_dist * leg + (1 - leg) * r_dist
        p_dist = jnp.linalg.norm(target - pp)
        hit = jnp.where( min_dist < 0.15, 1, 0)
        state.info["hit_time"] = ( state.info["hit_time"] + self.dt ) * hit

        khit = 0.9
        foot_rew = khit * jnp.exp(-1 * min_dist / 0.20)
        pelvis_rew = (1 - khit) * jnp.exp(-1 * p_dist / 0.5)
        rews =  foot_rew + pelvis_rew

        state.info["l_xy"] = lp[0:2]
        state.info["r_xy"] = rp[0:2]


        progress = jnp.where(state.info["hit_time"] > DS_TIME, 1, 0)

        state.info["hit_time"] = state.info["hit_time"] * (1 - progress)
        state.info["pointer"] = (jnp.roll(state.info["pointer"], 1) * progress +
                                 state.info["pointer"] * (1 - progress))

        state.info["fplan_reward"] += rews * 30
        return rews