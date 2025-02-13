from datetime import datetime
import functools
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from matplotlib import pyplot as plt
import dill
from nemo_env_pd import *
from networks.lstm import make_ppo_networks
from nemo_randomize import domain_randomize
import os

envs.register_environment('nemo', NemoEnv)
env = envs.get_environment('nemo')
eval_env = envs.get_environment('nemo')

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
      ppo.train, num_timesteps=180000000, num_evals=20, episode_length = 1000,
       normalize_observations=False, unroll_length=20, num_minibatches=64,
      num_updates_per_batch=4, discounting=0.995, learning_rate=3.0e-4,
      entropy_cost=1e-3, num_envs=2048, batch_size=1024,
      network_factory=make_networks_factory, randomization_fn = domain_randomize,
      )
#, restore_checkpoint_path=load_checkpoint_dir included notebook save_checkpoint_path=checkpoint_dir
x_data = []
y_data = {}
for name in metrics_dict.keys():
    y_data[name] = []
prefix = "eval/episode_"
times = [datetime.now()]

def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    for key in y_data.keys():
        y_data[key].append(metrics[prefix + key])
    plt.xlim([0, train_fn.keywords['num_timesteps']])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.title('{}'.format(metrics['eval/episode_reward']))
    for key in y_data.keys():
        num = float(metrics[prefix + key])
        plt.plot(x_data, y_data[key], label = key + " {:.2f}".format(num))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    make_inference_fn, params, _= train_fn(environment=env,
                                           progress_fn=progress,
                                           eval_env=eval_env)
    model.save_params("walk_policy", params)
    with open("inference_fn", 'wb') as f:
        dill.dump(make_inference_fn, f)
