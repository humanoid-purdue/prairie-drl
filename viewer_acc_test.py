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
import mediapy

OBS_SIZE = 334
ACT_SIZE = 24
DT = 0.01


mj_model = mujoco.MjModel.from_xml_path('nemo4b/scene.xml')
data = mujoco.MjData(mj_model)
viewer = mujoco.viewer.launch_passive(mj_model, data)
mj_model.opt.timestep = 0.001

#renderer = mujoco.Renderer(mj_model, width = 1920, height = 1080)

def get_sensor_data(sensor_name):
    sensor_id = mj_model.sensor(sensor_name).id
    sensor_adr = mj_model.sensor_adr[sensor_id]
    sensor_dim = mj_model.sensor_dim[sensor_id]
    return sensor_adr, sensor_dim

gyro = get_sensor_data("gyro_pelvis")
vel_p = get_sensor_data("local_linvel_pelvis")
acc_p = get_sensor_data("accelerometer_pelvis")

def _get_obs(data1, s_info):
    inv_pelvis_rot = math.quat_inv(data1.xquat[1])
    angvel = data1.sensordata[gyro[0]: gyro[0] + gyro[1]]
    vel = data1.sensordata[vel_p[0]: vel_p[0] + vel_p[1]]
    acc = data1.sensordata[acc_p[0]: acc_p[0] + acc_p[1]]

    grav_vec = math.rotate(jnp.array([0, 0, -1]), inv_pelvis_rot)

    acc = acc - grav_vec * 9.81

    position = data1.qpos[7:]
    velocity = data1.qvel[6:]
    phase = s_info["phase"]
    vel_target = s_info["vel_target"]
    angvel_target = s_info["angvel_target"]
    halt = s_info["halt"]
    carry = s_info["lstm_carry"]
    prev_action = s_info["prev_action"]
    cmd = jnp.array([vel_target[0], vel_target[1], angvel_target[0], halt])



    phase_clock = jnp.array([jnp.sin(phase[0]), jnp.cos(phase[0]),
                             jnp.sin(phase[1]), jnp.cos(phase[1])])

    print(vel, angvel, grav_vec, position, velocity, phase_clock, cmd)

    obs = jnp.concatenate([carry, acc,
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

frames = []

def tanh2Action(action: jnp.ndarray):
    pos_t = action[:ACT_SIZE//2]
    vel_t = action[ACT_SIZE//2:]

    bottom_limit = joint_limit[1:, 0] # - q_offset
    top_limit = joint_limit[1:, 1] # - q_offset
    vel_sp = vel_t * 10

    #pos_sp = ((pos_t + 1) * (top_limit - bottom_limit) / 2 + bottom_limit)
    pos_sp = pos_t * 1.0

    return jnp.concatenate([pos_sp, vel_sp])


make_inference_fn = makeIFN()
policy_path = 'walk_policy_acc5'
saved_params = model.load_params(policy_path)
inference_fn = make_inference_fn(saved_params)
jit_inference_fn = jax.jit(inference_fn)
state_info = {
    "halt": 0.,
    "phase": jnp.array([jnp.pi, 0]),
    "vel_target": jnp.array([0.4, 0]),
    "angvel_target": jnp.array([0.]),
    "prev_action": jnp.zeros(ACT_SIZE),
    "lstm_carry": jnp.zeros([HIDDEN_SIZE * DEPTH * 2]),
    "prev_pos": data.xpos[1],
}
init_qpos = mj_model.keyframe('stand').qpos
prev_data = data
data.qpos = init_qpos
data.ctrl = np.zeros([ACT_SIZE])
mujoco.mj_step(mj_model, data)
rng = jax.random.PRNGKey(0)
t = 0
walk_forward = True
pelvis_b_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, 'pelvis_back')
pelvis_f_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, 'pelvis_front')

trajectory = np.zeros([10000, 25])
for c in range(10000):
    if walk_forward:
        state_info["halt"] = 0.0
        state_info["angvel_target"] = jax.numpy.array([0.0])
        state_info["velocity_target"] = jax.numpy.array([0.4, 0.0])
        pp1 = data.site_xpos[pelvis_f_id]
        pp2 = data.site_xpos[pelvis_b_id]
        facing_vec = (pp1 - pp2)[0:2]
        facing_vec = facing_vec / jnp.linalg.norm(facing_vec)
        #state_info["angvel_target"] = jnp.array([facing_vec[1] * -2])
    if (c > 6000 and c < 7000):
        state_info["halt"] = 1.0
        state_info["phase"] = jnp.array([0, jnp.pi])
    if c % round(DT / mj_model.opt.timestep) == 0:
        obs = _get_obs(data, state_info)
        #print(obs[256:])
        act_rng, rng = jax.random.split(rng)
        t = time.time()
        ctrl, _ = jit_inference_fn(obs, act_rng)
        #print(time.time() - t)
        raw_action = ctrl[2 * HIDDEN_SIZE * DEPTH:]
        act = tanh2Action(state_info["prev_action"])
        #act = tanh2Action(raw_action)
        data.ctrl = act
        state_info["prev_action"] = raw_action
        state_info["lstm_carry"] = ctrl[:2 * HIDDEN_SIZE * DEPTH]

    #print(np.sum(np.abs(data.qfrc_actuator * data.qvel)))
    print(data.qfrc_actuator)
    jps = data.qpos[7:]
    jvs = data.qvel[6:]
    trajectory[c, 0] = t
    trajectory[c, 1:13] = jps
    trajectory[c, 13:25] = jvs

    state_info["phase"] += 2 * jnp.pi * mj_model.opt.timestep / 1.0
    state_info["phase"] = jnp.mod(state_info["phase"], jnp.pi * 2)
    mujoco.mj_step(mj_model, data)
    viewer.cam.trackbodyid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'l_hip_yaw')
    viewer.cam.distance = 1.5  # Distance from the target
    viewer.cam.lookat[:] = data.body("l_hip_yaw").xpos
    viewer.sync()
    t += mj_model.opt.timestep

    camera_id = mj_model.camera("cam")
    #renderer.update_scene(data)
    #pixels = renderer.render()
    #frame = np.copy(pixels)
    #frames.append(frame)
#renderer.close()
#mediapy.write_video('nemo_simulation.mp4', frames, fps=60)
#np.savetxt("nemo_traj.csv", trajectory, delimiter = ',')
viewer.close()
time.sleep(0.5)