import jax
from brax.envs import PipelineEnv, State
from jax import numpy as jnp
from brax.io import mjcf
import mujoco

metrics_dict = {
                    'reward': 0.0,
                    'lin_vel': 0.0,
                    'ang_vel': 0.0,
                    'ang_vel_xy': 0.0,
                    'orientation': 0.0,
                    'feet_air_time': 0.0,
                    'feet_slip': 0.0,
                    'feet_phase': 0.0,
                    'termination': 0.0,
                    'knee': 0.0,
                    'hip': 0.0,
                    'joint_limit': 0.0}

class NemoEnv(PipelineEnv):
    def __init__(self):
        model = mujoco.MjModel.from_xml_path("nemo/scene.xml")

        model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        model.opt.iterations = 6
        model.opt.ls_iterations = 6

        self.model = model

        system = mjcf.load_model(model)

        n_frames = 10

        super().__init__(sys = system,
            backend='mjx',
            n_frames = n_frames
        )

        self.initial_state = jnp.array(system.mj_model.keyframe('stand').qpos)
        self.nv = system.nv
        self.nu = system.nu
        self.control_range = system.actuator_ctrlrange
        self.joint_limit = jnp.array(model.jnt_range)

        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        self.pelvis_b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'pelvis_back')
        self.pelvis_f_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'pelvis_front')
        self.head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, 'head')

        self.left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_foot_roll')
        self.right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_foot_roll')

        self.left_foot_s1 = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "left_foot_p1")
        self.left_foot_s2 = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "left_foot_p2")
        self.left_foot_s3 = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "left_foot_p3")

        self.right_foot_s1 = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "right_foot_p1")
        self.right_foot_s2 = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "right_foot_p2")
        self.right_foot_s3 = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "right_foot_p3")

        self.floor_id = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.right_geom_id = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot")
        self.left_geom_id = mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot")


    def reset(self, rng: jax.Array):
        return

