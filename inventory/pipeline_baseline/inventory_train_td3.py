import numpy as np
import torch
import gym
import argparse
import os

from render.inventory_renderer import AsciiWorldRenderer
from env.inventory_env import InventoryManageEnv
from env.inventory_utils import Utils
from scheduler.inventory_random_policy import ProducerBaselinePolicy, BaselinePolicy
from scheduler.inventory_eoq_policy import ConsumerEOQPolicy
from scheduler.inventory_minmax_policy import ConsumerMinMaxPolicy as ConsumerBaselinePolicy
from utility.tools import SimulationTracker
from config.inventory_config import env_config

from tqdm import tqdm as tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument("--torch", action="store_true")
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--episod", type=int, default=60)
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--use-prev-action-reward", action="store_true")
parser.add_argument("--stop-iters", type=int, default=20)
parser.add_argument("--stop-timesteps", type=int, default=1e7)
parser.add_argument("--start-timesteps", type=int, default=1e5)
parser.add_argument("--stop-reward", type=float, default=150.0)
parser.add_argument("--pt", type=int, default=0)


def render(trainer, args, policy_mode):
# Create the environment
    evaluation_epoch_len = 60
    env.set_iteration(1, 1)
    env.discount_reward_training = False
    env.env_config.update({'episod_duration': evaluation_epoch_len})
    print(f"Environment: Producer action space {env.action_space_producer}, Consumer action space {env.action_space_consumer}, Observation space {env.observation_space}")
    
    policies = {}
    for agent_id in env.agent_ids():
        policies[agent_id] = trainer.get_policy(agent_id)

    # Simulation loop
    tracker = SimulationTracker(evaluation_epoch_len, 1, env, policies)
    
    if args.pt:
        loc_path = f"{os.environ['PT_OUTPUT_DIR']}/{policy_mode}/"
    else:
        loc_path = 'output/%s/' % policy_mode
    tracker.run_and_render(loc_path)


if __name__ == "__main__":
    args = parser.parse_args()
    env_config_for_rendering = env_config.copy()
    episod_duration = args.episod
    env_config_for_rendering['episod_duration'] = episod_duration
    env = InventoryManageEnv(env_config_for_rendering)

    # Create the environment
    env.set_iteration(1, 1)

    config = {
        "env_name": "SC",
        "seed": 0,
        "discount": 0.9,
        "tau": 5e-3,
        "time_steps": args.stop_timesteps,
        "start_time_step": args.start_timesteps,
        "expl_noise": 0.1,
        "batch_size": 256,
        "evaluate_frequency": 1e4
    }

    step_iters = args.stop_timesteps // 10
    trainer = TD3Trainer(config, env, False)
    for i in range(10):
        print('iteration % i ' % i)
        rewards = trainer.train(int(i*step_iters), int((i+1)*step_iters))
        render(trainer, args, 'td3_%i' % i )