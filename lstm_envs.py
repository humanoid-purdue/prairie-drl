from nemo_lstm import *

class Nemo4Env(NemoEnv):
    def __init__(self):
        super().__init__(rname = "nemo4")

class G2Env(NemoEnv):
    def __init__(self):
        super().__init__(rname = "g2")