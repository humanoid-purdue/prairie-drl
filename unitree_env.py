import jax
from brax.envs import PipelineEnv, State
from jax import numpy as jnp
import brax
from brax.io import mjcf
from brax.base import Base, Motion, Transform
from brax import math
import numpy as np
import mujoco

class UnitreeEnv(PipelineEnv):

    def __init__(self,
            obs_noise: float = 0.05,
            disturbance_vel: float = 0.05,
            contact_limit: float = 0.021,
            done_limit: float = 0.5,
            timestep: float = 0.025,
            action_scale: float = 0.5,
            **kwargs,):

        self.obs_noise = obs_noise
        self.disturbance_vel = disturbance_vel

        self.contact_limit = contact_limit
        self.done_limit = done_limit
        self.timestep = timestep
        self.action_scale = action_scale

        model = mujoco.MjModel.from_xml_path("unitree_g1/scene.xml")
        system = mjcf.load_model(model)

        n_frames = kwargs.pop('n_frames', 4)

        super().__init__(
            sys = system,
            backend='mjx',
            n_frames = n_frames
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

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        pipeline_state = self.pipeline_init(self.initial_state, jnp.zeros(self.nv))
        state_info = {
            "prev_torque": jnp.zeros(self.nu),
            "vel_command": self.control_commands(rng),
            "feet_air_time": jnp.zeros(2),
            "step": 0,
            "rng": rng,
            "step_total": 0,
            "distance": 0.0,
            "reward": 0.0
        }
        reward, done = jnp.zeros(2)
        metrics = {'distance': 0.0,
                   'reward': 0.0}

        obs = self.state2Obs(state_info, pipeline_state)
        state = State(
            pipeline_state = pipeline_state,
            obs = obs,
            reward = reward,
            done = done,
            metrics = metrics,
            info = state_info
        )
        return state

    def control_commands(
            self,
            rng: jax.Array,
    ) -> jax.Array:
        key1, key2, key3 = jax.random.split(rng, 3)
        velocity_x_limit = [0.0, 1.5]
        velocity_y_limit = [-0.5, 0.5]
        velocity_x_command = jax.random.uniform(key1, shape=(1,), minval=velocity_x_limit[0],
                                                maxval=velocity_x_limit[1])
        velocity_y_command = jax.random.uniform(key2, shape=(1,), minval=velocity_y_limit[0],
                                                maxval=velocity_y_limit[1])
        col_command = jnp.array([velocity_x_command[0], velocity_y_command[0]])
        return col_command

    def step(self, state: State, action: jax.Array) -> State:
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

        #get contact status and contact time for helping setup reward costs

        l_site_pos = pipeline_state.site_xpos[self.left_site]
        r_site_pos = pipeline_state.site_xpos[self.right_site]

        left_contact = l_site_pos[2] < self.contact_limit
        right_contact = r_site_pos[2] < self.contact_limit
        contact_arr = jnp.array([left_contact, right_contact])
        first_contact = (state.info['feet_air_time'] > 0) * contact_arr
        state.info['feet_air_time'] += self.timestep

        done = body_pos.pos[self.pelvis_id - 1, 2] < self.done_limit


        reward_linvel = self.rewardLinearVel(state.info, body_vel) * 2.0
        reward_zvel = self.rewardZVel(body_vel) * -1
        reward_angvel = self.rewardAngVel(body_vel) * -0.5
        reward_jt = self.rewardTorque(pipeline_state.qfrc_actuator) * -0.00005
        reward_dt = self.rewardDeltaTau(scaled_action, state.info["prev_torque"]) * -0.1
        reward_move = self.rewardMovement(state.info["vel_command"], pipeline_state.q) * -1
        reward_swing =  self.rewardSwing(state.info['feet_air_time'], first_contact, state.info["vel_command"]) * 8
        reward_single = self.rewardSingleSupport(contact_arr, state.info["vel_command"]) * 0.3
        reward_terminate = self.rewardTermination(done, state.info["step_total"]) * -1
        reward_jlimit = self.rewardJointLimit(pipeline_state.q) * -2
        reward_orien = self.rewardOrien(body_pos) * -1
        reward_cross = self.rewardCross(body_pos) * -2
        reward_pelvisz = self.rewardPelvisZ(body_pos) * -2
        #reward_yorein = self.rewardYOrien()


        state.info['feet_air_time'] *= ~contact_arr

        reward = (reward_linvel +
                  reward_zvel +
                  reward_angvel +
                  reward_jt +
                  reward_dt +
                  reward_move +
                  reward_swing +
                  reward_single +
                  reward_terminate +
                  reward_jlimit +
                  reward_orien +
                  reward_cross +
                  reward_pelvisz)

        state.info["rng"] = rng
        state.info["reward"] = reward
        state.info["prev_torque"] = scaled_action
        state.info['step'] += 1
        state.info["step_total"] += 1
        state.info['distance'] = math.normalize(body_pos.pos[self.pelvis_id - 1][:2])[1]

        state.info['vel_command'] = jnp.where(
            # condition: step>500
            state.info['step'] > 500,
            # if true
            self.control_commands(ctl_rng),
            # if false
            state.info['vel_command']
        )

        state.metrics['distance'] = state.info['distance']
        state.metrics['reward'] = reward
        # reset the step counter when the episode is terminated or reached 500 steps
        state.info['step'] = jnp.where(
            # condition: done or step>500
            done | (state.info['step'] > 500),
            # if true
            0,
            # if false
            state.info['step']
        )

        done = jnp.float32(done)
        # Wrap the state
        state = state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
        )


        return state

    def rewardLinearVel(self, state_info, body_vel):
        linear_vel_err = jnp.sum(jnp.square(state_info["vel_command"][:2] - body_vel.vel[self.pelvis_id-1][:2]))
        reward = jnp.exp(-linear_vel_err/0.25)
        return reward

    def rewardZVel(self, body_vel):
        return jnp.square(body_vel.vel[self.pelvis_id-1,2])

    def rewardAngVel(self, body_vel):
        return jnp.sum(jnp.square(body_vel.ang[self.pelvis_id-1,:2]))

    def rewardTorque(self, joint_torque):
        return jnp.sqrt(jnp.sum(jnp.square(joint_torque)))

    def rewardDeltaTau(self, action, prev_action):
        return jnp.mean((action - prev_action)**2)

    def rewardMovement(self, command, joint_angle):
        return jnp.mean((joint_angle - self.initial_state) ** 2) * (math.normalize(command[:2])[1] < 0.1)

    def rewardSwing(self, air_time, first_contact, vel_command):
        reward = jnp.sum((air_time - 0.3) * first_contact)
        reward *= (math.normalize(vel_command[:2])[1] > 0.1)
        return reward

    def rewardSingleSupport(self, contact, vel_command):
        singe_contact = jnp.sum(contact) == 1
        return singe_contact * (math.normalize(vel_command[:2])[1] > 0.1)

    def rewardTermination(self, done, step):
        terminal_early = done * (step < 950)
        reward = (950 - step) * terminal_early
        return reward

    def rewardJointLimit(self, joint_angle):
        limit = self.joint_limit * 0.95
        out_of_limit = -jnp.clip(joint_angle[6:] - limit[:, 0], max=0., min=None)
        out_of_limit += jnp.clip(joint_angle[6:] - limit[:, 1], max=None, min=0.)
        return jnp.sum(out_of_limit)

    def rewardOrien(self, body_pos):
        up = jnp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, body_pos.rot[self.pelvis_id - 1])
        reward = jnp.sum(jnp.square(rot_up[:2]))
        return reward

    def rewardCross(self, body_pos):
        global_y = jnp.array([0.0, 1.0, 0.0])
        local_y = math.rotate(global_y, math.quat_inv(body_pos.rot[self.pelvis_id - 1]))
        left_feet_pos = body_pos.pos[self.left_foot_id - 1] - body_pos.pos[self.pelvis_id - 1]
        right_feet_pos = body_pos.pos[self.right_foot_id - 1] - body_pos.pos[self.pelvis_id - 1]

        # ignore z-axis
        local_y = local_y[:2]
        left_feet_pos = left_feet_pos[:2]
        right_feet_pos = right_feet_pos[:2]

        # project the feet position to local y-axis
        left_feet_y = jnp.dot(left_feet_pos, local_y) / math.normalize(local_y)[0]
        right_feet_y = jnp.dot(right_feet_pos, local_y) / math.normalize(local_y)[0]

        # check its local position
        # left feet should have a positive value, right feet should have a negative value
        reward = left_feet_y[1] < 0
        reward |= right_feet_y[1] > 0

        return reward

    def rewardPelvisZ(self, body_pos):
        reward = jnp.abs(1.0 - body_pos.pos[self.pelvis_id - 1, 2])
        return reward

    def rewardYOrien(self, orientation, body_pos):
        global_x = jnp.array([1.0, 0.0, 0.0])

        # calculate the local y-axis
        local_x = math.rotate(global_x, body_pos.rot[self.pelvis_id - 1])

        # ignore z-axis
        local_x = local_x[:2]

        # calculate the error
        reward = jnp.sum(jnp.abs(orientation - local_x))

        return reward

    def state2Obs(self, state_info, pipeline_state):
        inv_pelvis_rot = math.quat_inv(pipeline_state.x.rot[self.pelvis_id - 1])
        grav_unit_vec = math.rotate(jnp.array([0,0,-1]),inv_pelvis_rot)

        pelvis_vel = pipeline_state.xd.vel[self.pelvis_id - 1]
        pelvis_angvel = pipeline_state.xd.ang[self.pelvis_id - 1]

        commanded_vel = state_info["vel_command"] #size 2 x y vel command

        prev_action = state_info["prev_torque"]

        joint_angle = pipeline_state.q
        joint_vel = pipeline_state.qd

        left_foot_pos = pipeline_state.xpos[self.left_foot_id].flatten() - pipeline_state.xpos[self.pelvis_id - 1].flatten()
        right_foot_pos = pipeline_state.xpos[self.left_foot_id].flatten() - pipeline_state.xpos[self.pelvis_id - 1].flatten()

        obs_vec = jnp.concatenate([inv_pelvis_rot, grav_unit_vec, pelvis_vel, pelvis_angvel,
                                   commanded_vel, prev_action, joint_angle, joint_vel,
                                   left_foot_pos, right_foot_pos])

        obs = obs_vec + self.obs_noise * jax.random.uniform(
            state_info['rng'], shape = obs_vec.shape, minval=-1, maxval=1)
        return obs
