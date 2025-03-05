import mujoco
import numpy as np
import math
import time
import mujoco.viewer
import jax.numpy as jnp
import jax
from brax import math
from networks.lstm import HIDDEN_SIZE, DEPTH
from brax.io import html, mjcf, model

OBS_SIZE = 334
ACT_SIZE = 24
DT = 0.035


mj_model = mujoco.MjModel.from_xml_path('nemo4/scene.xml')
data = mujoco.MjData(mj_model)
viewer = mujoco.viewer.launch_passive(mj_model, data)
mj_model.opt.timestep = 0.002


def get_sensor_data(sensor_name):
    sensor_id = mj_model.sensor(sensor_name).id
    sensor_adr = mj_model.sensor_adr[sensor_id]
    sensor_dim = mj_model.sensor_dim[sensor_id]
    return sensor_adr, sensor_dim

gyro = get_sensor_data("gyro_pelvis")
vel_t = get_sensor_data("local_linvel_pelvis")

def _get_obs(data1, state_info):
    inv_pelvis_rot = math.quat_inv(data1.xquat[1])
    angvel = data1.sensordata[gyro[0]: gyro[0] + gyro[1]]
    vel = data1.sensordata[vel_t[0]: vel_t[0] + vel_t[1]]

    grav_vec = math.rotate(jnp.array([0, 0, -1]), inv_pelvis_rot)
    position = data1.qpos
    velocity = data1.qvel
    phase = state_info["phase"]
    vel_target = state_info["vel_target"]
    angvel_target = state_info["angvel_target"]
    halt = state_info["halt"]
    carry = state_info["lstm_carry"]
    prev_action = state_info["prev_action"]
    cmd = jnp.array([vel_target[0], vel_target[1], angvel_target[0], halt])



    phase_clock = jnp.array([jnp.sin(phase[0]), jnp.cos(phase[0]),
                             jnp.sin(phase[1]), jnp.cos(phase[1])])

    obs = jnp.concatenate([carry, vel,
                           angvel, grav_vec, position, velocity, prev_action, phase_clock, cmd
                           ])
    return obs




def makeIFN():
    from brax.training.agents.ppo import networks as ppo_networks
    from networks.lstm import make_ppo_networks
    import functools
    from brax.training.acme import running_statistics
    mpn = make_ppo_networks
    network_factory = functools.partial(
        mpn,
        policy_hidden_layer_sizes=(512, 256, 256, 128))
    # normalize = running_statistics.normalize
    normalize = lambda x, y: x
    obs_size = OBS_SIZE
    ppo_network = network_factory(
        obs_size, ACT_SIZE, preprocess_observations_fn=normalize
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    return make_inference_fn

joint_limit = jnp.array(mj_model.jnt_range)

def tanh2Action(action: jnp.ndarray):
    pos_t = action[:ACT_SIZE//2]
    vel_t = action[ACT_SIZE//2:]

    bottom_limit = joint_limit[1:, 0] # - q_offset
    top_limit = joint_limit[1:, 1] # - q_offset
    vel_sp = vel_t * 10

    pos_sp = ((pos_t + 1) * (top_limit - bottom_limit) / 2 + bottom_limit)

    return jnp.concatenate([pos_sp, vel_sp])


make_inference_fn = makeIFN()
policy_path = 'walk_policy'
saved_params = model.load_params(policy_path)
inference_fn = make_inference_fn(saved_params)
jit_inference_fn = jax.jit(inference_fn)
state_info = {
    "halt": 0.,
    "phase": jnp.array([0, jnp.pi]),
    "vel_target": jnp.array([0.2, 0]),
    "angvel_target": jnp.array([0.]),
    "prev_action": jnp.zeros(ACT_SIZE),
    "lstm_carry": jnp.zeros([HIDDEN_SIZE * DEPTH * 2]),
    "prev_pos": data.xpos[1]
}
prev_data = data
data.ctrl = np.zeros([ACT_SIZE])
mujoco.mj_step(mj_model, data)
rng = jax.random.PRNGKey(0)
mj_model.geom_friction[0, 0] = 1.0
print(mj_model.geom_friction)
t = 0
for c in range(10000):
    if c % round(0.035 / mj_model.opt.timestep) == 0:
        obs = _get_obs(data, state_info)
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(obs, act_rng)
        raw_action = ctrl[2 * HIDDEN_SIZE * DEPTH:]
        state_info["prev_action"] = raw_action
        state_info["lstm_carry"] = ctrl[:2 * HIDDEN_SIZE * DEPTH]
        data.ctrl = tanh2Action(raw_action)
    state_info["phase"] += 2 * jnp.pi * mj_model.opt.timestep / 1.0
    state_info["phase"] = jnp.mod(state_info["phase"], jnp.pi * 2)
    mujoco.mj_step(mj_model, data)
    viewer.sync()
    t += mj_model.opt.timestep

viewer.close()
time.sleep(0.5)