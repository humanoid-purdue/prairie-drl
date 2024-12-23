import mujoco

model = mujoco.MjModel.from_xml_path("unitree_g1/scene.xml")

model.opt.solver = mujoco.mjtSolver.mjSOL_CG
model.opt.iterations = 6
model.opt.ls_iterations = 6

left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "collision")
for c in range(100):
    a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c)
    print(a)