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
