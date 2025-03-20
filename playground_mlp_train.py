from datetime import datetime
import functools
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from matplotlib import pyplot as plt
import dill
#from nemo_randomize import domain_randomize
from playground_randomize import domain_randomize
import os
from playground_envs import G1MLPEnv
from mujoco_playground import wrapper

def make_trainfns(robot = "g1"):
    if robot == "g1":
        env = G1MLPEnv()
    eval_env = env



    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 256, 128))


    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = os.path.join(os.path.abspath(os.getcwd()), checkpoint_dir)
        os.makedirs(checkpoint_dir)

    load_checkpoint_dir = 'load_checkpoints'
    if not os.path.exists(load_checkpoint_dir):
        load_checkpoint_dir = os.path.join(os.path.abspath(os.getcwd()), load_checkpoint_dir)
        load_checkpoint_dir = None


    train_fn = functools.partial(
        ppo.train, num_timesteps=200_000_000, num_evals=20, episode_length=1000,
        normalize_observations=False, unroll_length=20, num_minibatches=32,
        num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4,
        entropy_cost=0.005, num_envs=8192, batch_size=256, clipping_epsilon = 0.2,
        num_resets_per_eval = 1, action_repeat=1, max_grad_norm=1.0,
        reward_scaling = 1.0,
        network_factory=make_networks_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training
    )
    #, restore_checkpoint_path=load_checkpoint_dir included notebook save_checkpoint_path=checkpoint_dir
    x_data, y_data, y_dataerr = [], [], []
    times = [datetime.now()]
    prefix = "eval/episode_"
    times = [datetime.now()]

    def progress(num_steps, metrics):

        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        y_dataerr.append(metrics["eval/episode_reward_std"])

        plt.xlim([0, 100_000_000 * 1.25])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={y_data[-1]:.3f}")
        plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
        plt.show()
    return train_fn, env, progress, eval_env

if __name__ == "__main__":
    train_fn, env, progress, eval_env = make_trainfns("g1")
    make_inference_fn, params, _= train_fn(environment=env,
                                           progress_fn=progress,
                                           eval_env=eval_env)
    model.save_params("walk_policy", params)
    with open("inference_fn", 'wb') as f:
        dill.dump(make_inference_fn, f)