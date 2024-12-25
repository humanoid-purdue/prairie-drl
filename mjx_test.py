import mujoco

model = mujoco.MjModel.from_xml_path("unitree_g1/scene_pd.xml")

model.opt.solver = mujoco.mjtSolver.mjSOL_CG
model.opt.iterations = 6
model.opt.ls_iterations = 6

right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot")
left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot")
print(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, 11))
print(right_foot_id, left_foot_id)