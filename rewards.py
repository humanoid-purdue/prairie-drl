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
  right_bm = jnp.where(right_bm == 0, 1, 0)

  left_bm = jnp.sum(jnp.abs(dx.contact.geom - jnp.array([[floor_id, left_foot_id]])), axis=1)
  left_bm = jnp.where(left_bm == 0, 1, 0)

  # Sum forces for the identified contacts
  total_right_forces = jnp.sum(forces * right_bm[:, None], axis=0)
  total_left_forces = jnp.sum(forces * left_bm[:, None], axis=0)

  return total_left_forces, total_right_forces

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
    l_i = jnp.array([0., 0.117])
    r_i = jnp.array([0., -0.117])

    l_i2 = jnp.array([0., 0.11])
    r_i2 = jnp.array([0., -0.11])
    cycle_no = jnp.floor_divide(t, (ds_time + ss_time) * 2)


if __name__ == "__main__":
    v1 = []
    v2 = []
    for c in range(200):
        rewl, rewr = dualCycleCC(0.15, 0.4, 0.04, c / 100)
        v1 += [rewl]
        v2 += [rewr]
    import matplotlib.pyplot as plt
    plt.plot(v1)
    plt.plot(v2)
    plt.show()
