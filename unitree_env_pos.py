import jax
from brax.envs import PipelineEnv, State
from jax import numpy as jnp
import brax
from brax.io import mjcf
from brax.base import Base, Motion, Transform
from brax import math
import numpy as np
import mujoco
from brax import actuator

#Positional environment:
#Obeservation: robot joint pos and vel, robot centroid vels, Centroid positions relative to base, Command Velocity, Orientations, Facing Angle
#Initial task, fixed orien and command vel
#Base rewards include rewards for: alive, velocity reward, z vel, z pelvis, angvel, toruqe reward
#Periodic reward from https://arxiv.org/pdf/2011.01387 to help shape walk trajectory
# Expected value left * speed + Expected value right * speed
# Estimate force for other half: constant for if less than height, 0 if above height to approximate
# Initial periodic gait has no von mises, approximation with 1 - (2x - 1)^6
#
#to implement: code to prop hold upper body action to zero

class UnitreeEnvPos(PipelineEnv):
    def __init__(self, obs_noise: float = 0.05,
            disturbance_vel: float = 0.05,
            contact_limit: float = 0.051,
            done_limit: float = 0.4,
            timestep: float = 0.01,
            action_scale: float = 0.5,
            **kwargs):

        self.obs_noise = obs_noise
        self.disturbance_vel = disturbance_vel

        self.contact_limit = contact_limit
        self.done_limit = done_limit
        self.timestep = timestep
        self.action_scale = action_scale

        model = mujoco.MjModel.from_xml_path("unitree_g1/scene.xml")
        system = mjcf.load_model(model)

        n_frames = kwargs.pop('n_frames', 10)

        super().__init__(
            sys=system,
            backend='mjx',
            n_frames=n_frames
        )

        self.control_range = system.actuator_ctrlrange
        self.initial_state = jnp.array(system.mj_model.keyframe('stand').qpos)
        self.joint_limit = jnp.array(model.jnt_range)
        self.standing = system.mj_model.keyframe('stand').ctrl
        self.jnt_size = len(self.standing)
        self.nv = system.nv
        self.nu = system.nu

        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        self.left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_ankle_roll_link')
        self.right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_ankle_roll_link')
        self.left_site = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, 'left_foot')
        self.right_site = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, 'right_foot')

        return

    def control_commands(
            self,
            rng: jax.Array,
    ) -> jax.Array:
        key1, key2, key3 = jax.random.split(rng, 3)
        velocity_x_limit = [0.7, 0.8]
        velocity_y_limit = [-0.1, 0.1]
        velocity_x_command = jax.random.uniform(key1, shape=(1,), minval=velocity_x_limit[0],
                                                maxval=velocity_x_limit[1])
        velocity_y_command = jax.random.uniform(key2, shape=(1,), minval=velocity_y_limit[0],
                                                maxval=velocity_y_limit[1])
        col_command = jnp.array([velocity_x_command[0], velocity_y_command[0]])
        return col_command

    def state2Obs(self, state_info, pipeline_state):
        # Previous action
        # Pelvis quaternion
        # Pelvis z position
        # Joint positions
        # Joint velocities
        # Joint angvel
        # qfrc_actuator


        pelvis_rot = pipeline_state.x.rot[self.pelvis_id - 1]
        inv_pelvis_rot = math.quat_inv(pelvis_rot)
        grav_unit_vec = math.rotate(jnp.array([0,0,-1]),inv_pelvis_rot)

        pelvis_vel = pipeline_state.xd.vel[self.pelvis_id - 1]
        pelvis_angvel = pipeline_state.xd.ang[self.pelvis_id - 1]
        pelvis_z = pipeline_state.x.pos[self.pelvis_id - 1]

        commanded_vel = state_info["vel_command"] #size 2 x y vel command
        #command_orien_vec = state_info["orien_vec_cmd"] # size 2 xy orien unit vec

        qfrc_actuator = actuator.to_tau(
            self.sys, state_info["prev_torque"], pipeline_state.q, pipeline_state.qd)

        com, inertia, mass_sum, x_i = self._com(pipeline_state)
        cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        com_inertia = jnp.hstack(
            [cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]]
        )

        joint_angle = pipeline_state.q
        joint_vel = pipeline_state.qd

        left_foot_pos = pipeline_state.x.pos[self.left_foot_id].flatten() - pipeline_state.x.pos[self.pelvis_id - 1].flatten()
        right_foot_pos = pipeline_state.x.pos[self.left_foot_id].flatten() - pipeline_state.x.pos[self.pelvis_id - 1].flatten()

        obs_vec = jnp.concatenate([commanded_vel, pelvis_rot, grav_unit_vec, pelvis_vel, pelvis_angvel,
                                   joint_angle, joint_vel,
                                   left_foot_pos, right_foot_pos, pelvis_z, qfrc_actuator, com_inertia.ravel()])

        obs = obs_vec + self.obs_noise * jax.random.uniform(
            state_info['rng'], shape = obs_vec.shape, minval=-1, maxval=1)
        return obs

    def _com(self, pipeline_state) -> jax.Array:
        inertia = self.sys.link.inertia
        if self.backend in ['spring', 'positional']:
            inertia = inertia.replace(
                i=jax.vmap(jnp.diag)(
                    jax.vmap(jnp.diagonal)(inertia.i)
                    ** (1 - self.sys.spring_inertia_scale)
                ),
                mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
            )
        mass_sum = jnp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = (
                jnp.sum(jax.vmap(jnp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        )
        return com, inertia, mass_sum, x_i

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        pipeline_state = self.pipeline_init(self.initial_state, jnp.zeros(self.nv))

        state_info = {
            "prev_torque": jnp.zeros(self.nu),
            "vel_command": self.control_commands(rng),
            "rng": rng,
            "step_total": 0,
            "distance": 0.0,
            "reward": 0.0,
            "contact_state": jnp.array([1., 1.]),
            "air_time": 0.0
        }
        reward, done = jnp.zeros(2)
        metrics = {'distance': 0.0,
                   'reward': 0.0}

        obs = self.state2Obs(state_info, pipeline_state)
        state = State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=state_info
        )
        return state

    def step(self, state: State, action: jax.Array) -> State:

        #Replace periodic reward structure with one from dial mpc
        #Reward from  foot z position and target z position
        #Reward foot air time
        #Predicted position error reward
        #Reward upright
        #Reward 0 yaw
        #Reward z height

        rng, ctl_rng, disturb_rng = jax.random.split(state.info['rng'], 3)

        #rescale tanh to torque limits
        bottom_limit = self.control_range[:, 0]
        top_limit = self.control_range[:, 1]
        scaled_action = ( (action + 1) * (top_limit - bottom_limit) / 2 + bottom_limit ) * self.action_scale

        #scaled action is the torque commands given to the motors


        pipeline_state = self.pipeline_step(state.pipeline_state, scaled_action)

        body_pos = pipeline_state.x
        body_vel = pipeline_state.xd

        #Convert pipeline_state to an observation vector passable to the MLP

        obs = self.state2Obs(state.info, pipeline_state)
        done = body_pos.pos[self.pelvis_id - 1, 2] < self.done_limit

        time_elapsed = state.info["step_total"] * 0.01

        reward_linvel = self.rewardLinearVel(state.info, body_vel) * 2.0
        reward_footz = self.rewardFootZ(body_pos, time_elapsed) * 5.0
        reward_pos = self.rewardPos(body_pos, state.info) * 1.0
        reward_upright = self.rewardUpright(body_pos) * 0.5
        #reward_yaw = self.rewardYaw(body_pos)
        reward_air, cs, air_time = self.rewardAirtime(state.info, body_pos)
        reward_air = reward_air * 0.5
        reward_jt = self.rewardTorque(scaled_action) * -0.001
        reward_z = self.rewardPelvisZ(body_pos) * -2
        reward_term = done * -1500

        reward = reward_linvel + reward_footz + reward_pos + reward_upright + reward_air + reward_jt + reward_z + reward_term


        prop = jnp.mod(state.info["step_total"] * self.timestep, 0.8) / 0.8



        state.info["rng"] = rng
        state.info["reward"] = reward
        state.info["prev_torque"] = scaled_action
        state.info["step_total"] += 1
        state.info['distance'] = math.normalize(body_pos.pos[self.pelvis_id - 1][:2])[1]
        state.info["contact_state"] = cs
        state.info["air_time"] = air_time
        state.metrics['reward'] = reward
        done = jnp.float32(done)
        state = state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
        )


        return state

    def rewardLinearVel(self, state_info, body_vel):
        reward = -jnp.sum(jnp.square(state_info["vel_command"][:2] - body_vel.vel[self.pelvis_id-1][:2]))
        return reward

    def rewardTorque(self, joint_torque):
        return jnp.sqrt(jnp.sum(jnp.square(joint_torque)))

    def rewardPelvisZ(self, body_pos):
        reward = jnp.square(0.65 - body_pos.pos[self.pelvis_id - 1, 2])
        return reward

    def rewardFootZ(self, body_pos, time_elapsed):
        left_z = body_pos.pos[self.left_foot_id, 2]
        right_z = body_pos.pos[self.left_foot_id, 2]
        left_target, right_target = self.footstepZHeights(time_elapsed)
        reward_l = -1 * jnp.sum(jnp.square(left_target - left_z))
        reward_r = -1 * jnp.sum(jnp.square(right_target - right_z))
        return reward_l + reward_r

    def rewardPos(self, body_pos, state_info):
        linear_vel = state_info["vel_command"]
        xy_pos = body_pos.pos[self.pelvis_id - 1][0:2]
        expected_postiion = linear_vel * state_info["step_total"] * 0.01
        reward_pos = -jnp.sum(( xy_pos - expected_postiion) ** 2)
        return reward_pos

    def rewardUpright(self, body_pos):
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, body_pos.rot[self.pelvis_id - 1])
        return -jnp.sum(jnp.square(vec - vec_tar))

    def rewardYaw(self, body_pos):
        yaw = math.quat_to_euler(body_pos.rot[self.pelvis_id - 1])[2]
        reward_yaw = -jnp.square(jnp.atan2(jnp.sin(yaw), jnp.cos(yaw)))
        return reward_yaw

    def footstepZHeights(self, time_elapsed):
        z_height = 0.15
        swing_time = 0.5
        stance_time = 0.2
        def zCycle(time_elapsed):
            t2 = jnp.mod(time_elapsed, ( swing_time + stance_time ) * 2)
            swing_component = jnp.sin(time_elapsed) * z_height
            return jnp.where(t2 < swing_time, swing_component, 0)
        left_height = zCycle(time_elapsed - stance_time)
        right_height = zCycle(time_elapsed - stance_time * 2 - swing_time)
        return left_height, right_height

    def rewardAirtime(self, state_info, body_pos): #Sum airtime until contact is made and add the reward
        left_contact = body_pos.pos[self.left_foot_id][2] < 0.04
        right_contact = body_pos.pos[self.right_foot_id][2] < 0.04
        cs = jnp.array([left_contact * 1., right_contact * 1.])
        prev_cs = state_info["contact_state"]
        #When goes from [0, 1] to [1, 1], give reward. Any other transition remove reward
        single_contact = jnp.sum(cs - prev_cs == 1) == 1
        reward = single_contact * (state_info["air_time"] - 0.1)
        air_time = ( state_info["air_time"] + 0.01 ) * (jnp.sum(cs) != 2)
        return reward, cs, air_time