import jax
from jax import numpy as jnp
import brax
from brax.io import mjcf
import numpy as np
import mujoco


f = open("unitree_g1/g1.xml")
xml_str = f.read()
mj_model = mujoco.MjModel.from_xml_string(xml_str)
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.MjData(mj_model)

system = mjcf.load_model(mj_model)

print(system.actuator_ctrlrange)
print(system.mj_model.keyframe('stand').ctrl)

print(system.nv)
print(system.nu)

pelvis_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
left_foot_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'left_ankle_roll_link')
right_foot_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'right_ankle_roll_link')

print(pelvis_id)
print(left_foot_id)

print(len(system.actuator_ctrlrange))

