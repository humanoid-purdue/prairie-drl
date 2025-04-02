from nemo_lstm import *
import tomllib

class Nemo4Env(NemoEnv):
    def __init__(self):
        super().__init__(rname = "nemo4")

class Nemo4bEnv(NemoEnv):
    def __init__(self):
        super().__init__(rname = "nemo4b")

class G2Env(NemoEnv):
    def __init__(self):
        super().__init__(rname = "g2")
