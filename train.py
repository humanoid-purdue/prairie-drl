from datetime import datetime
import functools
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from matplotlib import pyplot as plt
import dill
import unitree_env_pd
from unitree_env_pd import UnitreeEnvMini

envs.register_environment('g1', UnitreeEnvMini)
env = envs.get_environment('g1')
eval_env = envs.get_environment('g1')

make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128))

pre_model_path = 'walk_policy'
pre_model = model.load_params(pre_model_path)

train_fn = functools.partial(
      ppo.train, num_timesteps=250000000, num_evals=40, episode_length = 2000,
       normalize_observations=True, unroll_length=20, num_minibatches=64,
      num_updates_per_batch=4, discounting=0.98, learning_rate=2.0e-4,
      entropy_cost=1e-3, num_envs=2048, batch_size=1024,
      network_factory=make_networks_factory)

x_data = []
y_data = {}
for name in unitree_env_pd.metrics_dict.keys():
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
    for key in y_data.keys():
        plt.plot(x_data, y_data[key], label = key)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    make_inference_fn, params, _= train_fn(environment=env,
                                           progress_fn=progress,
                                           eval_env=eval_env)
    model.save_params("walk_policy", params)
    with open("inference_fn", 'wb') as f:
        dill.dump(make_inference_fn, f)