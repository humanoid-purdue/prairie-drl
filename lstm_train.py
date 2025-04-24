from datetime import datetime
import functools
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from matplotlib import pyplot as plt
import dill
from lstm_envs import *
from networks.lstm import make_ppo_networks
from nemo_randomize import domain_randomize
import os
import tomllib

def make_trainfns(robot_file_path = "input_files/nemo4.toml"):
    class GenBotEnv(NemoEnv):
        def __init__(self):
            super().__init__(rfile_path = robot_file_path)
    
    envs.register_environment('nemo', GenBotEnv)
    
    env = envs.get_environment('nemo')
    eval_env = envs.get_environment('nemo')

    make_networks_factory = functools.partial(
        make_ppo_networks,
            policy_hidden_layer_sizes=(512, 256, 256, 128))


    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = os.path.join(os.path.abspath(os.getcwd()), checkpoint_dir)
        os.makedirs(checkpoint_dir)

    load_checkpoint_dir = 'load_checkpoints'
    if not os.path.exists(load_checkpoint_dir):
        load_checkpoint_dir = os.path.join(os.path.abspath(os.getcwd()), load_checkpoint_dir)
        load_checkpoint_dir = None

    with open(robot_file_path, "rb") as f:
            model_info = tomllib.load(f)

    model_train_func_parameters = model_info['train_func_parameters']

    train_func_num_timesteps_const = model_train_func_parameters['train_func_num_timesteps_const']
    train_func_num_evals_const = model_train_func_parameters['train_func_num_evals_const']
    train_func_episode_length_const = model_train_func_parameters['train_func_episode_length_const']
    train_func_normalize_observations_bool = model_train_func_parameters['train_func_normalize_observations_bool']
    train_func_unroll_length_const = model_train_func_parameters['train_func_unroll_length_const']
    train_func_num_minibatches_const = model_train_func_parameters['train_func_num_minibatches_const']
    train_func_num_updates_per_batch_const = model_train_func_parameters['train_func_num_updates_per_batch_const']
    train_func_discounting_const = model_train_func_parameters['train_func_discounting_const']
    train_func_learning_rate_const = model_train_func_parameters['train_func_learning_rate_const']
    train_func_entropy_cost_const = model_train_func_parameters['train_func_entropy_cost_const']
    train_func_num_envs_const = model_train_func_parameters['train_func_num_envs_const']
    train_func_batch_size_const = model_train_func_parameters['train_func_batch_size_const']
    train_func_clipping_epsilon_const = model_train_func_parameters['train_func_clipping_epsilon_const']
    train_func_num_resets_per_eval_const = model_train_func_parameters['train_func_num_resets_per_eval_const']
    train_func_action_repeat_const = model_train_func_parameters['train_func_action_repeat_const']
    train_func_max_grad_norm_const = model_train_func_parameters['train_func_max_grad_norm_const']
    train_func_reward_scaling_const = model_train_func_parameters['train_func_reward_scaling_const']

    train_fn = functools.partial(
        ppo.train, num_timesteps=300000000, num_evals=20, episode_length=1000,
        normalize_observations=False, unroll_length=20, num_minibatches=32,
        num_updates_per_batch=4, discounting=0.97, learning_rate=0.00005,
        entropy_cost=0.0005, num_envs=8192, batch_size=256, clipping_epsilon=0.2,
        num_resets_per_eval=1, action_repeat=1, max_grad_norm=1.0,
        reward_scaling=1.0,
        network_factory=make_networks_factory, randomization_fn=domain_randomize,
    )

    """
    train_fn = functools.partial(
        ppo.train, 
        num_timesteps=train_func_num_timesteps_const, 
        num_evals=train_func_num_evals_const, 
        episode_length=train_func_episode_length_const,
        normalize_observations=train_func_normalize_observations_bool, 
        unroll_length=train_func_unroll_length_const,
        num_minibatches=train_func_num_minibatches_const,
        num_updates_per_batch=train_func_num_updates_per_batch_const,
        discounting=train_func_discounting_const, 
        learning_rate=train_func_learning_rate_const,
        entropy_cost=train_func_entropy_cost_const, 
        num_envs=train_func_num_envs_const, 
        batch_size=train_func_batch_size_const,
        clipping_epsilon=train_func_clipping_epsilon_const,
        num_resets_per_eval=train_func_num_resets_per_eval_const,
        action_repeat=train_func_action_repeat_const,
        max_grad_norm=train_func_max_grad_norm_const,
        reward_scaling=train_func_reward_scaling_const,
        network_factory=make_networks_factory, randomization_fn=domain_randomize,
    )
    """
    
        
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
    return train_fn, env, progress, eval_env

if __name__ == "__main__":
    train_fn, env, progress, eval_env = make_trainfns(robot_file_path = "input_file/nemo4.toml")
    make_inference_fn, params, _= train_fn(environment=env,
                                           progress_fn=progress,
                                           eval_env=eval_env)
    model.save_params("walk_policy", params)
    with open("inference_fn", 'wb') as f:
        dill.dump(make_inference_fn, f)
