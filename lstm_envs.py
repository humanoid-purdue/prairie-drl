from nemo_lstm import *
import tomllib

class Nemo4bEnv(NemoEnv):
    def __init__(self):
        super().__init__(rname = "nemo4b")

