import numpy as np
import jax.numpy as jnp

def cycleContactCoeff(ds_time, ss_time, buffer_time, t):
    tmod = jnp.mod(t, ( ds_time + ss_time ) * 2)
    # if t is within ds
    rew_ds = jnp.ones(1)
    # if t is within ss
    rew_ss = jnp.zeros(1)
    # if t is within buffer ds to ss
    td2s = ( tmod - (ds_time - buffer_time) ) / (2 * buffer_time)
    rew_d2s = 0.5 - jnp.tanh(8 * td2s - 4) * 0.5
    #rew_d2s = 2
    # if t is within buffer ss to ds
    ts2d = ( tmod - (ds_time + ss_time - buffer_time) ) / (2 * buffer_time)
    rew_s2d = jnp.tanh(8 * ts2d - 4) * 0.5 + 0.5
    #rew_s2d = 3

    rew = (rew_ds * jnp.where(tmod < ds_time - buffer_time, 1, 0) +
           rew_ss * jnp.where(tmod > ds_time + buffer_time, 1, 0) * jnp.where(tmod < ds_time + ss_time - buffer_time, 1, 0) +
           rew_d2s * jnp.where(tmod >= ds_time - buffer_time, 1, 0) * jnp.where(tmod <= ds_time + buffer_time, 1, 0) +
           rew_s2d * jnp.where(tmod >= ds_time + ss_time - buffer_time, 1, 0) * jnp.where(tmod <= ds_time + ss_time + buffer_time, 1, 0) +
           rew_ds * jnp.where(tmod > ds_time + ss_time + buffer_time, 1, 0)
           )
    return rew

def dualCycleCC(ds_time, ss_time, buffer_time, t):
    left_cc = cycleContactCoeff(ds_time, ss_time, buffer_time, t + buffer_time)
    right_cc = cycleContactCoeff(ds_time, ss_time, buffer_time, t - ds_time - ss_time + buffer_time)
    return left_cc, right_cc

import mujoco

def get_feet_forces(m, dx, forces):
  # Identifiers for the floor, right foot, and left foot
  floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
  right_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "right_foot")
  left_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_foot")

  # Find contacts that involve both the floor and the respective foot
  # This assumes dx.contact.geom contains two entries per contact, one for each of the two contacting geometries
  right_bm = jnp.sum(jnp.abs(dx.contact.geom - jnp.array([[floor_id, right_foot_id]])), axis = 1)
  right_bm2 = jnp.sum(jnp.abs(dx.contact.geom - jnp.array([[right_foot_id, floor_id]])), axis=1)
  right_bm = jnp.where(right_bm == 0 , 1, 0)
  right_bm2 = jnp.where(right_bm2 == 0, 1, 0)

  right_bm = right_bm + right_bm2


  left_bm = jnp.sum(jnp.abs(dx.contact.geom - jnp.array([[floor_id, left_foot_id]])), axis=1)
  left_bm2 = jnp.sum(jnp.abs(dx.contact.geom - jnp.array([[left_foot_id, floor_id]])), axis=1)
  left_bm = jnp.where(left_bm == 0, 1, 0)
  left_bm2 = jnp.where(left_bm2 == 0, 1, 0)

  left_bm = left_bm + left_bm2

  # Sum forces for the identified contacts
  total_right_forces = jnp.sum(forces * right_bm[:, None], axis=0)
  total_left_forces = jnp.sum(forces * left_bm[:, None], axis=0)

  return total_left_forces, total_right_forces

def linkPlan(ds_time, ss_time, t):
    step_height = 0.2

    init_l = jnp.array([ 8.06828946e-07, 1.17871300e-01, 0.0])
    init_r = jnp.array([ 8.06828946e-07, -1.17871300e-01, 0.0])
    init_c = jnp.array([0., 0., 0.75])

    def dsCycle(t):
        tmod = jnp.mod(t, 2 * (ss_time + ds_time))
        tprop = tmod / ss_time
        v1 = 4 * step_height * (tprop) * (1 - tprop)
        return jnp.where(tmod < ss_time, v1, 0)

    left_z = dsCycle( t - ds_time)
    right_z = dsCycle( t - (ds_time * 2 + ss_time))

    l_pos = init_l + jnp.array([0, 0, left_z[0]])
    r_pos = init_r + jnp.array([0, 0, right_z[0]])

    c_pos = jnp.array([0., 0., 0.55])
    return l_pos, r_pos, c_pos

def get_contact_forces(s, d):
    assert (s.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL)  # Assert cone is PYRAMIDAL

    # mju_decodePyramid
    # 1: force: result
    # 2: pyramid: d.efc_force + contact.efc_address
    # 3: mu: contact.friction
    # 4: dim: contact.dim

    contact = d.contact
    cnt = d.ncon

    # Generate 2d array of efc_force indexed by efc_address containing the maximum
    # number of potential elements (10).
    # This enables us to operate on each contact force pyramid rowwise.
    efc_argmap = jnp.linspace(
        contact.efc_address,
        contact.efc_address + 9,
        10, dtype=jnp.int32
    ).T
    # OOB access clamps in jax, this is safe
    pyramid = d.efc_force[efc_argmap.reshape((efc_argmap.size))].reshape(efc_argmap.shape)

    # Calculate normal forces
    # force[0] = 0
    # for (int i=0; i < 2*(dim-1); i++) {
    #   force[0] += pyramid[i];
    # }
    index_matrix = jnp.repeat(jnp.arange(10)[None, :], cnt, axis=0)
    force_normal_mask = index_matrix < (2 * (contact.dim - 1)).reshape((cnt, 1))
    force_normal = jnp.sum(jnp.where(force_normal_mask, pyramid, 0), axis=1)

    # Calculate tangent forces
    # for (int i=0; i < dim-1; i++) {
    #   force[i+1] = (pyramid[2*i] - pyramid[2*i+1]) * mu[i];
    # }
    pyramid_indexes = jnp.arange(5) * 2
    force_tan_all = (pyramid[:, pyramid_indexes] - pyramid[:, pyramid_indexes + 1]) * contact.friction
    force_tan = jnp.where(pyramid_indexes < contact.dim.reshape((cnt, 1)), force_tan_all, 0)

    # Full force array
    forces = jnp.concatenate((force_normal.reshape((cnt, 1)), force_tan), axis=1)

    # Special case frictionless contacts
    # if (dim == 1) {
    #   force[0] = pyramid[0];
    #   return;
    # }
    frictionless_mask = contact.dim == 1
    frictionless_forces = jnp.concatenate((pyramid[:, 0:1], jnp.zeros((pyramid.shape[0], 5))), axis=1)
    return jnp.where(
        frictionless_mask.reshape((cnt, 1)),
        frictionless_forces,
        forces
    )

def makeFootStepPlan(ds_time, ss_time, t):
    #determine number of steps forward.
    #swing right foot first
    # at each ds to ss transition iterate forward
    #initial_pos:
    def moduloFootstep(t):
        count = jnp.floor_divide(t, (2 * (ss_time + ds_time)))
        return count
    step_size = 0.2
    l_i = jnp.array([0., 0.117])
    r_i = jnp.array([0., -0.117])

    l_i2 = jnp.array([0., 0.11])
    r_i2 = jnp.array([-0.1, -0.11])

    right_count = moduloFootstep(t - ds_time) + 1
    left_count = moduloFootstep(t - ds_time * 2 - ss_time) + 1

    procedural_l_pos = l_i2 + left_count * jnp.array([step_size, 0.])
    procedural_r_pos = r_i2 + right_count * jnp.array([step_size, 0.])


    l_pos = jnp.where(t <= ds_time, l_i, procedural_l_pos)
    r_pos = jnp.where(t <= ds_time, r_i, procedural_r_pos)

    return l_pos, r_pos

class FootstepPlan:
    def __init__(self, ds_time, ss_time, buffer_time):
        self.ds_time = ds_time
        self.ss_time = ss_time
        self.buffer_time = buffer_time
        #footstep plan consists of an array of n by 4 array
        #Each footstep represented by 4 numbers, first 2 are pos, last 2 are direction vector
        c = 20
        step_size = 0.30
        self.left_plan = np.zeros([c, 4])
        self.right_plan = np.zeros([c, 4])
        self.left_plan[0, :] = np.array([0., 0.117, 1.0, 0.])
        self.right_plan[0, :] = np.array([0., -0.117, 1.0, 0.])
        for i in range(c - 1):
            l_x = (i // 2) * step_size + step_size / 2
            r_x = ((i + 1) // 2) * step_size
            self.left_plan[i + 1, :] = np.array([l_x, 0.1, 1.0, 0.0])
            self.right_plan[i + 1, :] = np.array([r_x, -0.1, 1.0, 0.0])

        self.left_plan = jnp.array(self.left_plan)
        self.right_plan = jnp.array(self.right_plan)

        self.bottom_limit = jnp.arange(c) * (self.ds_time + self.ss_time) - self.ss_time
        self.top_limit = (jnp.arange(c) + 1) * (self.ds_time + self.ss_time) - self.ss_time

        final_pos = ( self.left_plan[-1, :2] + self.right_plan[-1, :2] ) / 2
        final_time = ( self.bottom_limit[-1] + self.top_limit[-1] ) / 2
        self.vel = final_pos / final_time

    def getNumInfo(self, t):
        #Function gets the current footstep number and determines which legs are ground contact
        step_no = jnp.int32(jnp.floor_divide(t + self.ss_time, self.ds_time + self.ss_time))
        #Determine which legs are contacting ground
        l_cc, r_cc = dualCycleCC(self.ds_time, self.ss_time, self.buffer_time, t)
        return step_no, l_cc, r_cc

    def getStepInfo(self, t):
        weight_vec = jnp.where( self.bottom_limit < t, 1, 0) * jnp.where( self.top_limit > t, 1, 0)
        l_step = jnp.sum(self.left_plan * weight_vec[:, None], axis = 0)
        r_step = jnp.sum(self.right_plan * weight_vec[:, None], axis = 0)
        l_cc, r_cc = dualCycleCC(self.ds_time, self.ss_time, self.buffer_time, t)
        return l_step, r_step, l_cc, r_cc

if __name__ == "__main__":
    ds_time = 0.15
    ss_time = 0.4
    fsp = FootstepPlan(ds_time, ss_time, 0.05)
    l_x = []
    l_y = []
    for c in range(200):
        l, r, lcc, rcc = fsp.getStepInfo(c / 100)
        l_x += [lcc]
        l_y += [rcc]

    import matplotlib.pyplot as plt
    plt.plot(l_x)
    plt.plot(l_y)
    plt.show()
