
import dill
from IPython.display import HTML
import jax
from brax import envs
from brax.io import html, mjcf, model
from nemo_env_pd import NemoEnv
import mujoco
import mujoco.mjx as mjx

model = mujoco.MjModel.from_xml_path("nemo4b/scene.xml")

model.opt.solver = mujoco.mjtSolver.mjSOL_CG
model.opt.iterations = 6
model.opt.ls_iterations = 6

system = mjcf.load_model(model)

print(dir(model))
print(model.opt.timestep)
lknee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'l_knee')
rknee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'r_knee')
print(lknee_id, rknee_id)