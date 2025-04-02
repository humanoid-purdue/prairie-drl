from typing import Any, Dict, Optional, Union

from etils import epath
import mujoco

import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
import os

class BaseEnv(mjx_env.MjxEnv):
  """Base class for G1 environments."""

  def __init__(
      self,
      consts,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    self.consts = consts
    super().__init__(config, config_overrides)
    self._mj_model = mujoco.MjModel.from_xml_path(
        xml_path
    )
    self._mj_model.opt.timestep = self.sim_dt

    if self._config.restricted_joint_range:
      self._mj_model.jnt_range[1:] = consts.RESTRICTED_JOINT_RANGE
      self._mj_model.actuator_ctrlrange[:] = consts.RESTRICTED_JOINT_RANGE

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model)
    self._xml_path = xml_path

  # Sensor readings.

  def get_gravity(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the gravity vector in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{self.consts.GRAVITY_SENSOR}_{frame}"
    )

  def get_global_linvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the linear velocity of the robot in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{self.consts.GLOBAL_LINVEL_SENSOR}_{frame}"
    )

  def get_global_angvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the angular velocity of the robot in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{self.consts.GLOBAL_ANGVEL_SENSOR}_{frame}"
    )

  def get_local_linvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the linear velocity of the robot in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{self.consts.LOCAL_LINVEL_SENSOR}_{frame}"
    )

  def get_accelerometer(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the accelerometer readings in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{self.consts.ACCELEROMETER_SENSOR}_{frame}"
    )

  def get_gyro(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the gyroscope readings in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{self.consts.GYRO_SENSOR}_{frame}"
    )

  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
