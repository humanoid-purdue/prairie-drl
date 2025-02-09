import jax
from brax.envs import PipelineEnv, State
from jax import numpy as jnp
from brax.io import mjcf
from brax import math
import mujoco

import rewards


DS_TIME = 0.1
SS_TIME = 0.4
BU_TIME = 0.05
STEP_HEIGHT = 0.11


metrics_dict = {
                   'reward': 0.0,
                   'flatfoot': 0.0,
                   'periodic': 0.0,
                    'upright': 0.0,
                    'limit': 0.0,
                    'swing_height': 0.0,
                    'termination': 0.0,
                    'velocity': 0.0,
                    'energy': 0.0,
                    'angvel_xy': 0.0,
                    'action_rate': 0.0,
                    'vel_z': 0.0,
                    'feet_slip': 0.0,
                    'angvel_z': 0.0,
                    'feet_orien': 0.0,
                    'feet_slip_ang': 0.0,
                    'energy_symmetry': 0.0 }

class NemoEnv(PipelineEnv):
    def __init__(self):
        model = mujoco.MjModel.from_xml_path("nemo2/scene.xml")

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

        self.floor_id = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.right_geom_id = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot")
        self.left_geom_id = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot")


    def _get_obs_fk(
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

        velocity = data1.qvel * 0.05

        #l_grf, r_grf = self.determineGRF(data1)
        # external_contact_forces are excluded
        angvel = data1.xd.ang[self.pelvis_id, :] * 0.25

        com0 = data1.subtree_com[0]
        com1 = data1.subtree_com[1]
        com_vel = (com1 - com0) / self.dt
        com_vel = com_vel * 2

        z = data1.x.pos[self.pelvis_id, 2:3]

        if state is not None:
            t = state.info["time"]
            rng = state.info["rng"]

            rng, key = jax.random.split(rng)
            sites_noise_0 = jax.random.uniform(key, shape = prev_sites.shape, minval = -0.1, maxval = 0.1)
            prev_sites += sites_noise_0

            rng, key = jax.random.split(rng)
            sites_noise_1 = jax.random.uniform(key, shape = prev_sites.shape, minval=-0.1, maxval=0.1)
            current_sites += sites_noise_1

            rng, key = jax.random.split(rng)
            position_noise = jax.random.uniform(key, shape = position.shape, minval = -0.1, maxval = 0.1)
            position += position_noise

            rng, key = jax.random.split(rng)
            velocity_noise = jax.random.uniform(key, shape = velocity.shape, minval = -0.5, maxval = 0.5)
            velocity += velocity_noise * 0.05

            rng, key = jax.random.split(rng)
            angvel_noise = jax.random.uniform(key, shape = angvel.shape, minval = -0.4, maxval = 0.4)
            angvel += angvel_noise * 0.25

            rng, key = jax.random.split(rng)
            vel_noise = jax.random.uniform(key, shape = com_vel.shape, minval = -0.1, maxval = 0.1)
            com_vel += vel_noise * 2.0

            state.info["rng"] = rng

            vel_target = state.info["velocity"]
            angvel_target = state.info["angvel"]
            cmd = jnp.array([vel_target[0], vel_target[1], angvel_target[0]])

        else:
            t = 0
            cmd = jnp.array([0, 0, 0.])

        l_coeff, r_coeff = rewards.dualCycleCC(DS_TIME, SS_TIME, BU_TIME, t)

        return jnp.concatenate([
            position,
            velocity,
            angvel,
            com_vel,
            prev_sites, current_sites,
            prev_action, l_coeff, r_coeff, z, cmd
        ])

    def _get_obs(self, data0, data1, prev_action: jnp.ndarray, state = None):
        inv_pelvis_rot = math.quat_inv(data1.x.rot[self.pelvis_id - 1])
        vel = data1.xd.vel[self.pelvis_id - 1] * 2.0
        angvel = data1.xd.ang[self.pelvis_id - 1] * 0.25
        grav_vec = math.rotate(jnp.array([0,0,-1]),inv_pelvis_rot)
        position = data1.qpos
        velocity = data1.qvel * 0.05
        z = data1.x.pos[self.pelvis_id, 2:3]
        if state is not None:
            t = state.info["time"]
            rng = state.info["rng"]

            rng, key = jax.random.split(rng)
            position_noise = jax.random.uniform(key, shape = position.shape, minval = -0.05, maxval = 0.05)
            position += position_noise

            rng, key = jax.random.split(rng)
            velocity_noise = jax.random.uniform(key, shape = velocity.shape, minval = -0.5, maxval = 0.5)
            velocity += velocity_noise * 0.05

            rng, key = jax.random.split(rng)
            angvel_noise = jax.random.uniform(key, shape = angvel.shape, minval = -0.4, maxval = 0.4)
            angvel += angvel_noise * 0.25

            rng, key = jax.random.split(rng)
            grav_vec_noise = jax.random.uniform(key, shape = grav_vec.shape, minval = -0.1, maxval = 0.1)
            grav_vec += grav_vec_noise
            state.info["rng"] = rng
        else:
            t = 0.
        l_coeff, r_coeff = rewards.dualCycleCC(DS_TIME, SS_TIME, BU_TIME, t)

        obs = jnp.concatenate([
            vel, angvel, grav_vec, position, velocity, prev_action, l_coeff, r_coeff, z
        ])

        return obs

    def reset(self, rng: jax.Array) -> State:
        vel, angvel, rng = self.makeCmd(rng)
        pipeline_state = self.pipeline_init(self.initial_state, jnp.zeros(self.nv))

        state_info = {
            "rng": rng,
            "time": jnp.zeros(1),
            "velocity": vel,
            "angvel": angvel,
            "prev_action": jnp.zeros(self.nu),
            "energy_hist": jnp.zeros([100, 12]),
        }
        metrics = metrics_dict.copy()

        obs = self._get_obs_fk(pipeline_state, pipeline_state, jnp.zeros(self.nu))
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

    def makeCmd(self, rng):
        rng, key1 = jax.random.split(rng)
        rng, key2 = jax.random.split(rng)

        vel = jax.random.uniform(key1, shape=[2], minval = -1, maxval = 1)
        vel = vel * jnp.array([0.3, 0.3])
        #vel = vel + jnp.array([0.2, 0.0])
        angvel = jax.random.uniform(key2, shape=[1], minval=-0.7, maxval=0.7)
        return vel, angvel, rng

    def updateCmd(self, state):
        rng = state.info["rng"]
        vel, angvel, rng = self.makeCmd(rng)
        state.info["rng"] = rng
        tmod = jnp.mod(state.info["time"], 5.0)
        reroll_cmd = jnp.where(tmod > 4.98, 1, 0)
        state.info["velocity"] = state.info["velocity"] * (1 - reroll_cmd) + vel * reroll_cmd
        state.info["angvel"] = state.info["angvel"] * (1 - reroll_cmd) + angvel * reroll_cmd
        return

    def tanh2Action(self, action: jnp.ndarray):
        pos_t = action[:self.nu//2]
        vel_t = action[self.nu//2:]

        bottom_limit = self.joint_limit[1:, 0]
        top_limit = self.joint_limit[1:, 1]
        vel_sp = vel_t * 10
        pos_sp = ((pos_t + 1) * (top_limit - bottom_limit) / 2 + bottom_limit)

        return jnp.concatenate([pos_sp, vel_sp])
    #return pos_sp

    def updateEnergyHistory(self, state, data):
        qfrc_actuator = data.qfrc_actuator
        jv = data.qvel
        energy = qfrc_actuator * jv[6:] * self.dt
        state.info["energy_hist"] = jnp.concatenate([energy, state.info["energy_hist"][:][:99]])
        return

    def step(self, state: State, action: jnp.ndarray):
        scaled_action = self.tanh2Action(action)

        #apply noise to scaled action
        pos_action = scaled_action[scaled_action.shape[0]//2:]
        vel_action = scaled_action[scaled_action.shape[0]//2:]

        rng = state.info["rng"]
        rng, key = jax.random.split(rng)
        pos_noise = jax.random.uniform(key, shape = pos_action.shape, minval = -0.1, maxval = 0.1)
        pos_action += pos_noise

        rng, key = jax.random.split(rng)
        vel_noise = jax.random.uniform(key, shape = vel_action.shape, minval = -1.0, maxval = 1.0)
        vel_action += vel_noise

        state.info["rng"] = rng

        scaled_action = jnp.concatenate([pos_action, vel_action])

        data0 = state.pipeline_state
        data1 = self.pipeline_step(data0, scaled_action)

        contact = rewards.feet_contact(data1, self.floor_id, self.left_geom_id, self.right_geom_id)
        reward, done = self.rewards(state, data1, action, contact)

        state.info["time"] += self.dt
        state.info["prev_action"] = action
        self.updateCmd(state)

        self.updateEnergyHistory(state, data1)

        obs = self._get_obs_fk(data0, data1, action, state = state)
        return state.replace(
            pipeline_state = data1, obs=obs, reward=reward, done=done
        )

    def rewards(self, state, data, action, contact):
        reward_dict = {}
        data0 = state.pipeline_state
        min_z, max_z = (0.4, 0.7)
        is_healthy = jnp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(data.q[2] > max_z, 0.0, is_healthy)
        #healthy_reward = 1.2 * is_healthy
        #reward_dict["healthy"] = healthy_reward
        reward_dict["termination"] = -500 * (1 - is_healthy)

        vel_reward = self.velocityReward(state, data0, data)
        reward_dict["velocity"] = vel_reward * 2.0

        angvel_z_reward = self.angvelZReward(state, data)
        reward_dict["angvel_z"] = angvel_z_reward * 2.0

        angvel_xy_reward = self.angvelXYReward(data)
        reward_dict["angvel_xy"] = angvel_xy_reward * -0.15

        vel_z_reward = self.velZReward(data0, data)
        reward_dict["vel_z"] = vel_z_reward * -0.01

        energy_reward = self.energyReward(data)
        reward_dict["energy"] = energy_reward * -0.001

        action_r_reward = self.actionRateReward(action, state)
        reward_dict["action_rate"] = action_r_reward * -0.01

        upright_reward = self.uprightReward(data)
        reward_dict["upright"] = upright_reward * 1.5

        slip_reward = self.feetSlipReward(data0, data, contact)
        reward_dict["feet_slip"] = slip_reward * -0.25

        period_rew = self.periodicReward(state.info, data0, data)
        reward_dict["periodic"] = period_rew * 2.0

        limit_reward = self.jointLimitReward(data)
        reward_dict["limit"] = limit_reward * 5.0

        flatfoot_reward = self.flatfootReward(data, contact)
        reward_dict["flatfoot"] = flatfoot_reward * 2.0

        swing_height_reward = self.swingHeightReward(state.info, data)
        #reward_dict["swing_height"] = swing_height_reward * 100.0
        reward_dict["swing_height"] = swing_height_reward * 2.0

        feet_orien_reward = self.footOrienReward(data)
        reward_dict["feet_orien"] = feet_orien_reward * 1.0

        angslip_reward = self.feetSlipAngReward(data, contact)
        reward_dict["feet_slip_ang"] = angslip_reward * -0.25

        energy_symmetry_reward = self.energySymmetryReward(state.info)
        reward_dict["energy_symmetry"] = energy_symmetry_reward

        for key in reward_dict.keys():
            reward_dict[key] *= self.dt

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
        return jnp.exp( vel_n * -1 / 0.25)

    def angvelZReward(self, state, data):
        angvel = data.xd.ang[self.pelvis_id][2]
        angvel_err = jnp.square(angvel - state.info["angvel"][0])
        return jnp.exp(angvel_err * -1 / 0.25)

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
        pelvis_xy = ((data.site_xpos[self.pelvis_f_id] + data.site_xpos[self.pelvis_b_id]) / 2)[0:2]
        head_xy = data.site_xpos[self.head_id][0:2]
        xy_err = jnp.linalg.norm(pelvis_xy - head_xy)
        return jnp.exp(xy_err * -1 / 0.1)

    def energyReward(self, data):
        qfrc_actuator = data.qfrc_actuator
        jv = data.qvel
        energy = jnp.sum(jnp.square(jv * qfrc_actuator)) ** 0.5
        return energy

    def feetSlipReward(self, data0, data1, contact):
        lp0, rp0 = self.footPos(data0)
        lp1, rp1 = self.footPos(data1)

        lv = (lp1 - lp0) / self.dt
        rv = (rp1 - rp0) / self.dt

        feet_v = jnp.array([jnp.sum(jnp.square(lv)), jnp.sum(jnp.square(rv))])
        rew = feet_v * contact
        return jnp.sum(rew)

    def feetSlipAngReward(self, data1, contact):
        langvel = data1.xd.ang[self.left_foot_id][2]
        rangvel = data1.xd.ang[self.right_foot_id][2]

        feet_v = jnp.array([jnp.sum(jnp.square(langvel)), jnp.sum(jnp.square(rangvel))])
        rew = 0.3 * feet_v * contact
        return jnp.sum(rew)

    def periodicReward(self, info, data1, data0):
        t = info["time"]

        l_coeff, r_coeff = rewards.dualCycleCC(DS_TIME, SS_TIME, BU_TIME, t)

        l_contact_coeff = 2 * l_coeff -1
        r_contact_coeff = 2 * r_coeff - 1

        gnd_vel_coeff = -1
        swing_vel_coeff = 1
        l_vel_coeff = swing_vel_coeff - l_coeff * (swing_vel_coeff - gnd_vel_coeff)
        r_vel_coeff = swing_vel_coeff - r_coeff * (swing_vel_coeff - gnd_vel_coeff)

        l_grf, r_grf = self.determineGRF(data1)
        #l_nf = jnp.linalg.norm(l_grf[0:3])
        #r_nf = jnp.linalg.norm(r_grf[0:3])
        l_f_rew = 1 - jnp.exp(-1 * jnp.sum(l_grf[0:2] ** 2) / 100)
        r_f_rew = 1 - jnp.exp(-1 * jnp.sum(r_grf[0:2] ** 2) / 100)

        lp0, rp0 = self.footPos(data0)
        lp1, rp1 = self.footPos(data1)

        lv = (lp1 - lp0) / self.dt
        rv = (rp1 - rp0) / self.dt

        l_spd_rew = 1 - jnp.exp(-2 * jnp.sum(lv**2))
        r_spd_rew = 1 - jnp.exp(-2 * jnp.sum(rv**2))

        vel_reward = l_vel_coeff * l_spd_rew + r_vel_coeff * r_spd_rew
        grf_reward = l_contact_coeff * l_f_rew + r_contact_coeff * r_f_rew

        return (vel_reward + grf_reward)[0]

    def determineGRF(self, data):

        forces = rewards.get_contact_forces(self.model, data)
        lfoot_grf, rfoot_grf = rewards.get_feet_forces(self.model, data, forces)

        return lfoot_grf, rfoot_grf

    def flatfootReward(self, data, contact):
        vec_tar = jnp.array([0.0, 0.0, 1.0])

        def sites2Rew(p1, p2, p3):
            v1 = p1 - p3
            v2 = p2 - p3
            dot = jnp.cross(v1, v2)
            normal_vec = dot / jnp.linalg.norm(dot)
            ca = jnp.abs(normal_vec[2])
            reward = jnp.exp(-1 * (ca -1) ** 2 / 0.001)
            return reward

        lp1 = data.site_xpos[self.left_foot_s1]
        lp2 = data.site_xpos[self.left_foot_s2]
        lp3 = data.site_xpos[self.left_foot_s3]

        rp1 = data.site_xpos[self.right_foot_s1]
        rp2 = data.site_xpos[self.right_foot_s2]
        rp3 = data.site_xpos[self.right_foot_s3]

        rew = sites2Rew(lp1, lp2, lp3) + sites2Rew(rp1, rp2, rp3)

        return rew

    def jointLimitReward(self, data1):
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

        #being below is -1, being above is 0.
        l_rew = jnp.exp(jnp.clip(l_h - l_t, min = None, max = 0) / 0.02) - 1
        r_rew = jnp.exp(jnp.clip(r_h - r_t, min = None, max = 0) / 0.02) - 1

        #l_err = jnp.exp(-1 * jnp.sum(jnp.square(l_h - l_t)) / 0.001)
        #r_err = jnp.exp(-1 * jnp.sum(jnp.square(r_h - r_t)) / 0.001)

        rew = l_rew * (1 - l_coeff) + r_rew * (1 - r_coeff)
        return rew[0]

    def energySymmetryReward(self, state_info):
        leftEnergy = (jnp.sum(state_info["energy_hist"][:6][:]))
        rightEnergy = jnp.sum(state_info["energy_hist"][6:][:])
        difference = jnp.abs(leftEnergy-rightEnergy)

        return jnp.exp(-1*difference)

    def footOrienReward(self, data):
        lp1 = data.site_xpos[self.left_foot_s1]
        lp2 = data.site_xpos[self.left_foot_s2]

        rp1 = data.site_xpos[self.right_foot_s1]
        rp2 = data.site_xpos[self.right_foot_s2]

        l_vec = (lp1 - lp2)[0:2]
        l_vec = l_vec / jnp.linalg.norm(l_vec)
        r_vec = (rp1 - rp2)[0:2]
        r_vec = r_vec / jnp.linalg.norm(r_vec)

        pp1 = data.site_xpos[self.pelvis_f_id]
        pp2 = data.site_xpos[self.pelvis_b_id]
        facing_vec = (pp1 - pp2)[0:2]
        facing_vec = facing_vec / jnp.linalg.norm(facing_vec)

        dpl = jnp.sum(facing_vec * l_vec)
        dpr = jnp.sum(facing_vec * r_vec)

        l_rew = jnp.exp(-(dpl - 1)**2 / 0.1)
        r_rew = jnp.exp(-(dpr - 1) ** 2 / 0.1)
        return l_rew + r_rew
