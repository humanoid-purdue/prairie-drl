import jax
from jax import numpy as jnp
import brax
from brax.io import mjcf
import mujoco


f = open("pal_talos/talos.xml")
xml_str = f.read()
mj_model = mujoco.MjModel.from_xml_string(xml_str)
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.MjData(mj_model)

system = mjcf.load_model(mj_model)
print(system.actuator_ctrlrange)