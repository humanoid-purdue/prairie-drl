import mujoco

model = mujoco.MjModel.from_xml_path("unitree_g1/scene_pd.xml")

model.opt.solver = mujoco.mjtSolver.mjSOL_CG
model.opt.iterations = 6
model.opt.ls_iterations = 6
