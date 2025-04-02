from playground.joystick_mlp_env import Joystick
from unitree_g1.g1_consts import G1_CONSTS
import os

class G1MLPEnv(Joystick):
    def __init__(self):
        super().__init__(G1_CONSTS(), xml_path = os.getcwd() +
                                               "/unitree_g1/scene.xml")