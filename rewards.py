import numpy as np
import jax.numpy as jnp

from typing import Any, Tuple, Union

import jax
from mujoco import mjx

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

def heightLimit(ds_time, ss_time, buffer_time, step_height, t):
    def cycleQuad(ds_time, ss_time, t):
        tmod = jnp.mod(t, (ds_time + ss_time) * 2)
        peak = jnp.cos( (tmod - (ds_time + ss_time * 0.5) ) * jnp.pi / ss_time) * step_height
        sec_1 = jnp.where(tmod < ds_time, 0, 1)
        sec_2 = jnp.where(tmod > ds_time + ss_time, 0, 1)
        return peak * sec_1 * sec_2
    left_peak = cycleQuad(ds_time, ss_time, t + buffer_time)
    right_peak = cycleQuad(ds_time, ss_time, t - ds_time - ss_time + buffer_time)
    return left_peak, right_peak

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

def get_collision_info(
    contact: Any, geom1: int, geom2: int
) -> Tuple[jax.Array, jax.Array]:
  """Get the distance and normal of the collision between two geoms."""
  mask = (jnp.array([geom1, geom2]) == contact.geom).all(axis=1)
  mask |= (jnp.array([geom2, geom1]) == contact.geom).all(axis=1)
  idx = jnp.where(mask, contact.dist, 1e4).argmin()
  dist = contact.dist[idx] * mask[idx]
  normal = (dist < 0) * contact.frame[idx, 0, :3]
  return dist, normal


def geoms_colliding(state: mjx.Data, geom1: int, geom2: int) -> jax.Array:
  """Return True if the two geoms are colliding."""
  return get_collision_info(state.contact, geom1, geom2)[0] < 0

def feet_contact(state, floor_id, right_foot_id, left_foot_id):
    l = geoms_colliding(state, left_foot_id, floor_id)
    r = geoms_colliding(state, right_foot_id, floor_id)
    contact = jnp.array([l, r])
    return contact


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
        c = 40

        self.bottom_limit = jnp.arange(c) * (self.ds_time + self.ss_time) - self.ss_time - self.buffer_time
        self.top_limit = (jnp.arange(c) + 1) * (self.ds_time + self.ss_time) - self.ss_time - self.buffer_time


    def getStepInfo(self, left_plan, right_plan, t):
        weight_vec = jnp.where( self.bottom_limit < t, 1, 0) * jnp.where( self.top_limit > t, 1, 0)
        l_step = jnp.sum(left_plan * weight_vec[:, None], axis = 0)
        r_step = jnp.sum(right_plan * weight_vec[:, None], axis = 0)
        return l_step, r_step

def naiveFootstepPlan(ds_time, ss_time):
    l_steps = jnp.array([[0., 0.117]])
    r_steps = jnp.array([[0., -0.117]])
    step_size = 0.30
    c = 40
    for i in range(c - 1):
        l_x = (i // 2) * step_size + step_size / 2
        r_x = ((i + 1) // 2) * step_size
        l_steps = jnp.concatenate([l_steps, jnp.array([[l_x, 0.10]])], axis = 0)
        r_steps = jnp.concatenate([r_steps, jnp.array([[r_x, -0.10]])], axis = 0)
    ave_vel = (l_steps[-1, :] + r_steps[-1, :]) / ((c - 1) * (2 * ds_time + ss_time))
    return l_steps, r_steps, ave_vel

#swing left leg first
def sequentialFootstepPlan():
    l_y = 0.15
    r_y = -0.15
    step_size = 0.35
    steps = jnp.array([[step_size * 0.5, l_y]])
    pointer = jnp.zeros([80])
    pointer = jnp.concatenate([jnp.array([1]), pointer])

    weights = jnp.ones([78])
    weights = jnp.concatenate([jnp.array([1., 1., 1.]), weights])
    leg = jnp.array([1]) # 1 for left 0 for right
    for i in range(40):
        l_next = jnp.array([[step_size * i + step_size * 1.5, l_y]])
        r_next = jnp.array([[step_size * i + step_size, r_y]])

        steps = jnp.concatenate([steps, r_next, l_next], axis = 0)
        leg = jnp.concatenate([leg, jnp.array([0]), jnp.array([1])], axis = 0)

    return steps, pointer, weights, leg

def get_rz(
    phi: Union[jax.Array, float], swing_height: Union[jax.Array, float] = 0.08
) -> jax.Array:
  def cubic_bezier_interpolation(y_start, y_end, x):
    y_diff = y_end - y_start
    bezier = x**3 + 3 * (x**2 * (1 - x))
    return y_start + y_diff * bezier

  x = (phi + jnp.pi) / (2 * jnp.pi)
  stance = cubic_bezier_interpolation(0, swing_height, 2 * x)
  swing = cubic_bezier_interpolation(swing_height, 0, 2 * x - 1)
  return jnp.where(x <= 0.5, stance, swing)

#Phase is a 0 - 2pi value for each foot offset by pi radians.
def lr_phase_coeff(phase, ds_prop, bu_prop):
    #buffer lies entirely within ds prop

    def phase_sol(t):
        ds_d = jnp.pi * ds_prop
        bu_d = ds_d * bu_prop

        stance = 0.0
        swing = 1.0

        def norm_p(x, st):
            return (x - st) / bu_d

        h00 = lambda x: 2 * x**3 - 3 * x**2 + 1
        h01 = lambda x: -2 * x**3 + 3 * x**2

        t2s = lambda x: h01(norm_p(x, ds_d - bu_d))
        s2t = lambda x: h00(norm_p(x, np.pi - ds_d))

        p1 = jnp.where(t <= ds_d - bu_d, 1, 0)
        p2 = jnp.where(t <= ds_d, 1, 0) * jnp.where( ds_d - bu_d < t, 1, 0)
        p3 = jnp.where(ds_d < t, 1, 0) * jnp.where(t <= np.pi - ds_d, 1, 0)
        p4 = jnp.where(np.pi - ds_d < t, 1, 0) * jnp.where(t <= np.pi - ds_d + bu_d, 1, 0)
        p5 = jnp.where(t > np.pi - ds_d + bu_d, 1, 0)
        return (p1 * stance +
                p2 * t2s(t) +
                p3 * swing +
                p4 * s2t(t) +
                p5 * stance)
    l = phase_sol(phase[0])
    r = phase_sol(phase[1])
    return jnp.array([l, r])



def quintic_foot_phase(phase, ds_prop):
    def phase_sol(t):
        coeffs = jnp.array([0.1 , 5.0, -18.8, 12.0, 9.6])
        ds_d = jnp.pi * ds_prop
        nt = (t - ds_d) / (2 * (np.pi - ds_d * 2))
        z2 = (coeffs[0] * nt +
             coeffs[1] * nt**2 +
             coeffs[2] * nt**3 +
             coeffs[3] * nt**4 +
             coeffs[4] * nt**5)
        zd2 = (coeffs[0] +
              coeffs[1] * 2 * nt +
              coeffs[2] * 3 * nt**2 +
              coeffs[3] * 4 * nt**3 +
              coeffs[4] * 5 * nt**4)
        p2 = jnp.where(t > ds_d, 1, 0) * jnp.where(t <= np.pi - ds_d, 1, 0)
        return p2 * z2, p2 *zd2
    lz, lzd = phase_sol(phase[0])
    rz, rzd = phase_sol(phase[1])
    z_h = jnp.array([lz, rz])
    zd_h = jnp.array([lzd, rzd])
    return z_h, zd_h

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    phase = jnp.array([0, jnp.pi])
    lr = jnp.zeros([1000, 2])
    lrz = jnp.zeros([1000, 2])
    for c in range(1000):
        lr = lr.at[c, :].set(lr_phase_coeff( phase,0.1, 0.5))
        z, zd = quintic_foot_phase(phase, 0.1)
        lrz = lrz.at[c, :].set(z)
        phase += 0.02
        phase = jnp.mod(phase, jnp.pi * 2)
    plt.plot(lrz[:, 0])
    plt.plot(lrz[:, 1])
    plt.show()
