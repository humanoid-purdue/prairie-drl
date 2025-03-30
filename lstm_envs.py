from nemo_lstm import *
import tomllib

class Nemo4Env(NemoEnv):
    def __init__(self):
        super().__init__(rname = "nemo4")

class G2Env(NemoEnv):
    def __init__(self):
        super().__init__(rname = "g2")

class GenBotEnv(NemoEnv):
    def __init__(self, input_toml_file_path):
        with open(input_toml_file_path, "rb") as f:
            model_info = tomllib.load(f)
        model_general = model_info["general"]

        super().__init__(rname = model_general["model_name"])
        self.toml_file_path = input_toml_file_path
