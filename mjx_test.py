import mujoco
from brax.io import mjcf
import mujoco.mjx as mjx
import jax.numpy as jnp

model = mujoco.MjModel.from_xml_path("unitree_g1/scene_pd.xml")
a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
model.opt.solver = mujoco.mjtSolver.mjSOL_CG
model.opt.iterations = 6
model.opt.ls_iterations = 6
system = mjcf.load_model(model)
mj_data = mujoco.MjData(model)
#mj_data = mj_data.replace(ctrl = [0.1] * 13)
#mj_data.ctrl = [0.5] * 13
print(mj_data.ctrl)
print(mj_data.act)
print(dir(mj_data))

for c in range(100):
    mj_data.ctrl = [0] * 13
    mj_data.act = [1] * 13
    mujoco.mj_step(model, mj_data)
    print(mj_data.ctrl, mj_data.act, mj_data.qpos)