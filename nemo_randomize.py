#returns a function handle for randomization_fn. copy domain_randomizer from playground

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


def domain_randomize(model: mjx.Model, rng: jax.Array):
  @jax.vmap
  def rand_dynamics(rng):
    # Floor friction: =U(0.4, 1.0).
    rng, key = jax.random.split(rng)
    floor_id = 0
    geom_friction = model.geom_friction.at[floor_id, 0].set(
        jax.random.uniform(key, minval=0.4, maxval=1.0)
    )

    # Scale static friction: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
        key, shape=(12,), minval=0.9, maxval=1.1
    )
    dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

    # Scale armature: *U(1.0, 1.05).
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[6:] * jax.random.uniform(
        key, shape=(12,), minval=1.0, maxval=1.05
    )
    dof_armature = model.dof_armature.at[6:].set(armature)

    # Scale all link masses: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(model.nbody,), minval=0.9, maxval=1.1
    )
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

    # Add mass to torso: +U(-1.0, 1.0).
    #rng, key = jax.random.split(rng)
    #dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
    #body_mass = body_mass.at[TORSO_BODY_ID].set(
    #    body_mass[TORSO_BODY_ID] + dmass
    #)

    # Jitter qpos0: +U(-0.05, 0.05).
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[7:].set(
        qpos0[7:]
        + jax.random.uniform(key, shape=(12,), minval=-0.05, maxval=0.05)
    )


    #Random facing vector
    rng, key = jax.random.split(rng)
    z = jax.random.uniform(key, shape = [1], minval = -1, maxval = 1)[0]
    w = ( 1 - z**2 ) ** 0.5
    quat = jnp.array([w, 0., 0., z])
    qpos0 = qpos0.at[3:7].set(quat)

    # Randomize timestep
    rng, key = jax.random.split(rng)
    timestep = jax.random.uniform(key, minval=0.0025, maxval=0.005)

    return (
        geom_friction,
        dof_frictionloss,
        dof_armature,
        body_mass,
        qpos0,
        timestep,
    )

  (
      friction,
      frictionloss,
      armature,
      body_mass,
      qpos0,
      timestep,
  ) = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes_opt = jax.tree_util.tree_map(lambda x: None, model.opt)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "qpos0": 0,
      "opt": in_axes_opt.tree_replace({"timestep": 0}),
  })

  model = model.tree_replace({
      "geom_friction": friction,
      "dof_frictionloss": frictionloss,
      "dof_armature": armature,
      "body_mass": body_mass,
      "qpos0": qpos0,
      "opt": model.opt.tree_replace({"timestep": timestep}),
  })

  return model, in_axes