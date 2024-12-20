import mujoco
from brax.io import mjcf

model = mujoco.MjModel.from_xml_path("unitree_g1/scene.xml")

model.opt.solver = mujoco.mjtSolver.mjSOL_CG
model.opt.iterations = 6
model.opt.ls_iterations = 6

print(model.jnt_range)
print(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis'))