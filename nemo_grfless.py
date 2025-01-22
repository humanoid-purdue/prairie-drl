import jax
from brax.envs import PipelineEnv, State
from jax import numpy as jnp
from brax.io import mjcf
import mujoco
from mujoco import mjx

import rewards
from jax import random

DS_TIME = 0.15
SS_TIME = 0.5
BU_TIME = 0.05
MIN_AT = 0.2
STEP_HEIGHT = 0.10

metrics_dict = {
                   'reward': 0.0,
                    'upright': 0.0,
                    'healthy': 0.0,
                    'velocity': 0.0,
                    'feet_airtime': 0.0,
                    'feet_clearance': 0.0,
                    'feet_phase': 0.0,
                    'feet_slip': 0.0,
                    'action_rate': 0.0,
                    'angvel_z': 0.0,
                    'angvel_xy': 0.0,
                    'vel_z': 0.0,
                    'energy': 0.0}

class NemoEnv(PipelineEnv):
    def __init__(self):
        model = mujoco.MjModel.from_xml_path("nemo/scene.xml")

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
        self.pelvis_b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'pelvis_back')
        self.pelvis_f_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'pelvis_front')
        self.head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, 'head')

        self.left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_foot_roll')
        self.right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_foot_roll')

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
        if state is None:
            t = 0
        else:
            t = state.info["time"]

        l_coeff, r_coeff = rewards.dualCycleCC(DS_TIME, SS_TIME, BU_TIME, t)
        l_t, r_t = rewards.heightLimit(DS_TIME, SS_TIME, BU_TIME, STEP_HEIGHT, t)

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
            prev_action, l_coeff, r_coeff
        ])

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        pipeline_state = self.pipeline_init(self.initial_state, jnp.zeros(self.nv))

        state_info = {
            "rng": rng,
            "time": jnp.zeros(1),
            "feet_airtime": jnp.zeros(2),
            "last_contact": jnp.array([0, 0]),
            "prev_action": jnp.zeros(self.nu),
            "velocity": jnp.array([0.5, 0]),
            "angvel": 0.0
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


        forces = rewards.get_contact_forces(self.model, data1)
        lfoot_grf, rfoot_grf = rewards.get_feet_forces(self.model, data1, forces)

        l_contact = jnp.where(jnp.linalg.norm(lfoot_grf) > 10, 1, 0)
        r_contact = jnp.where(jnp.linalg.norm(rfoot_grf) > 10, 1, 0)

        contact = jnp.array([l_contact, r_contact])
        contact_filt = contact | state.info["last_contact"]

        reward, done = self.reward(state, data1, action, contact)

        state.info["time"] += self.dt
        state.info["feet_airtime"] += self.dt
        state.info["feet_airtime"] *= ~contact
        state.info["last_contact"] = contact_filt
        state.info["prev_action"] = action

        obs = self._get_obs(data0, data1, action, state = state)
        return state.replace(
            pipeline_state = data1, obs=obs, reward=reward, done=done
        )

    def reward(self, state, data, action, contact):
        reward_dict = {}
        data0 = state.pipeline_state
        min_z, max_z = (0.5, 1.1)
        is_healthy = jnp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(data.q[2] > max_z, 0.0, is_healthy)
        healthy_reward = 1.2 * is_healthy
        reward_dict["healthy"] = healthy_reward

        vel_reward = self.velocityReward(state, data0, data)
        reward_dict["velocity"] = vel_reward * 1.0

        angvel_z_reward = self.angvelZReward(state, data)
        reward_dict["angvel_z"] = angvel_z_reward * 0.5

        angvel_xy_reward = self.angvelXYReward(data)
        reward_dict["angvel_xy"] = angvel_xy_reward * -0.15

        vel_z_reward = self.velZReward(data0, data)
        reward_dict["vel_z"] = vel_z_reward * -0.001

        energy_reward = self.energyReward(data)
        reward_dict["energy"] = energy_reward * -0.001

        action_r_reward = self.actionRateReward(action, state)
        reward_dict["action_rate"] = action_r_reward * -0.01

        upright_reward = self.uprightReward(data)
        reward_dict["upright"] = upright_reward * 2.0

        phase_reward = self.feetPhaseReward(state.info, data)
        reward_dict["feet_phase"] = phase_reward * 1.0

        air_time_reward = self.feetAirtime(state, contact)
        reward_dict["feet_airtime"] = air_time_reward * 2.0

        slip_reward = self.feetSlipReward(data0, data, contact)
        reward_dict["feet_slip"] = slip_reward * -0.25

        clearance_reward = self.feetClearanceReward(data0, data)
        reward_dict["feet_clearance"] = clearance_reward * 0.01


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

    def pelVel(self, data0, data1):
        pos0 = data0.x.pos[self.pelvis_id]
        pos1 = data1.x.pos[self.pelvis_id]
        vel = ( pos1 - pos0 ) / self.dt
        return vel

    def footPos(self, data):
        lp1 = data.site_xpos[self.left_foot_s1]
        lp2 = data.site_xpos[self.left_foot_s2]
        lp3 = data.site_xpos[self.left_foot_s3]

        rp1 = data.site_xpos[self.right_foot_s1]
        rp2 = data.site_xpos[self.right_foot_s2]
        rp3 = data.site_xpos[self.right_foot_s3]

        lp = (lp1 + lp2 + lp3) / 3
        rp = (rp1 + rp2 + rp3) / 3
        return lp, rp

    def velocityReward(self, state, data0, data1):
        vel = self.pelVel(data0, data1)
        vel_target = state.info["velocity"]
        vel_n = jnp.sum(jnp.square(vel[0:2] - vel_target))
        return jnp.exp( vel_n * -1 / 0.5)

    def angvelZReward(self, state, data):
        angvel = data.xd.ang[self.pelvis_id][2]
        angvel_err = jnp.square(angvel - state.info["angvel"])
        return jnp.exp(angvel_err * -1 / 0.5)

    def actionRateReward(self, action, state):
        act_delta = jnp.sum(jnp.square(state.info["prev_action"] - action))
        return jnp.exp(act_delta * -1)

    def angvelXYReward(self, data):
        angvel = data.xd.ang[self.pelvis_id][0:2]
        angvel_err = jnp.sum(jnp.square(angvel))
        return jnp.exp(angvel_err * -1)

    def velZReward(self, data0, data1):
        vel = self.pelVel(data0, data1)
        return jnp.square(vel[2])

    def uprightReward(self, data):
        body_pos = data.x
        pelvis_xy = body_pos.pos[self.pelvis_id][0:2]
        head_xy = data.site_xpos[self.head_id][0:2]
        xy_err = jnp.linalg.norm(pelvis_xy - head_xy)
        return jnp.exp(xy_err * -30)

    def energyReward(self, data):
        qfrc_actuator = data.qfrc_actuator
        jv = data.qvel
        energy = jnp.sum(jnp.square(jv * qfrc_actuator)) ** 0.5
        return energy

    def feetPhaseReward(self, info, data):
        t = info["time"]
        l_t, r_t = rewards.heightLimit(DS_TIME, SS_TIME, BU_TIME, STEP_HEIGHT, t)

        l_p, r_p = self.footPos(data)
        l_h = l_p[2]
        r_h = r_p[2]


        #l_rew = jnp.clip(l_h - l_t, min = -10, max = 0)
        #r_rew = jnp.clip(r_h - r_t, min = -10, max = 0)
        l_err = jnp.square(l_h - l_t)
        r_err = jnp.square(r_h - r_t)

        #rew = l_rew * (1 - l_coeff) + r_rew * (1 - r_coeff)
        rew = jnp.exp(-1 * (l_err + r_err) / 0.01)
        return rew[0]

    def feetClearanceReward(self, data0, data1):
        lp0, rp0 = self.footPos(data0)
        lp1, rp1 = self.footPos(data1)

        lv = (lp1 - lp0) / self.dt
        rv = (rp1 - rp0) / self.dt

        lh_e = (lp1[2] - STEP_HEIGHT) ** 2
        rh_e = (rp1[2] - STEP_HEIGHT) ** 2

        l_rew = lh_e * jnp.linalg.norm(lv[0:2]) ** 0.5
        r_rew = rh_e * jnp.linalg.norm(rv[0:2]) ** 0.5

        return l_rew + r_rew

    def feetAirtime(self, state, contact_filt):
        #airtime given when current contact state changes and airtime greater than zero
        first_contact = ( state.info["feet_airtime"] > 0.0 ) * contact_filt
        air_time = (state.info["feet_airtime"] - MIN_AT) * first_contact
        air_time = jnp.clip(air_time, max = SS_TIME - MIN_AT)
        rew = jnp.sum(air_time)
        return rew

    def feetSlipReward(self, data0, data1, contact):
        lp0, rp0 = self.footPos(data0)
        lp1, rp1 = self.footPos(data1)

        lv = (lp1 - lp0) / self.dt
        rv = (rp1 - rp0) / self.dt

        feet_v = jnp.array([jnp.sum(jnp.square(lv)), jnp.sum(jnp.square(rv))])
        rew = feet_v * contact
        return jnp.sum(rew)