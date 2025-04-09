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

    with open(robot_file_path, "rb") as f:
        model_info = tomllib.load(f)
    
    class GenBotEnv(NemoEnv):
        def __init__(self):
            super().__init__(model_info = model_info)
    
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

        train_fn = functools.partial(
        ppo.train, 
        num_timesteps =             model_info['train_func_parameters']['num_timesteps'], 
        num_evals =                 model_info['train_func_parameters']['num_evals'], 
        episode_length =            model_info['train_func_parameters']['episode_length'],
        normalize_observations =    model_info['train_func_parameters']['normalize_observations'], 
        unroll_length =             model_info['train_func_parameters']['unroll_length'],
        num_minibatches =           model_info['train_func_parameters']['num_minibatches'],
        num_updates_per_batch =     model_info['train_func_parameters']['num_updates_per_batch'],
        discounting =               model_info['train_func_parameters']['discounting'], 
        learning_rate =             model_info['train_func_parameters']['learning_rate'],
        entropy_cost =              model_info['train_func_parameters']['entropy_cost'], 
        num_envs =                  model_info['train_func_parameters']['num_envs'], 
        clipping_epsilon =          model_info['train_func_parameters']['clipping_epsilon'],
        batch_size =                model_info['train_func_parameters']['batch_size'],
        num_resets_per_eval =       model_info['train_func_parameters']['num_resets_per_eval'],
        action_repeat =             model_info['train_func_parameters']['action_repeat'],
        max_grad_norm =             model_info['train_func_parameters']['max_grad_norm'],
        reward_scaling =            model_info['train_func_parameters']['reward_scaling'],
        
        network_factory=make_networks_factory, randomization_fn=domain_randomize,
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
    return train_fn, env, progress, eval_env

if __name__ == "__main__":
    train_fn, env, progress, eval_env = make_trainfns(robot_file_path = "input_file/nemo4.toml")
    make_inference_fn, params, _= train_fn(environment=env,
                                           progress_fn=progress,
                                           eval_env=eval_env)
    model.save_params("walk_policy", params)
    with open("inference_fn", 'wb') as f:
        dill.dump(make_inference_fn, f)
