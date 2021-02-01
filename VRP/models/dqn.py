from models.replay_memory import replay_memory
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
from ray.rllib.policy import Policy
from ray.rllib.models.torch.misc import SlimFC, normc_initializer as \
    torch_normc_initializer

        

class DQNModule(nn.Module):
    def __init__(self, num_states=4, num_actions=18):
        """
        Initialize a deep Q-learning network for testing algorithm
            in_features: number of features of input.
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQNModule, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        layers = []
        prev_layer_size = num_states

        self.num_actions = num_actions

        # Create layers 0 to second-last.
        self.hidden_out_size = 64
        for size in [512, 256, 128, self.hidden_out_size]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=torch_normc_initializer(1.0),
                    activation_fn=nn.ReLU))
            prev_layer_size = size
            # Add a batch norm layer.
            # layers.append(nn.BatchNorm1d(prev_layer_size))

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=num_actions,
            initializer=torch_normc_initializer(1.0),
            activation_fn=None)

        self._hidden_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)
        self._hidden_out = self._hidden_layers(x)
        return self._value_branch(self._hidden_out)
        # action_mask = x[:, (-self.num_actions+1):]
        # value = self._value_branch(self._hidden_out)
        # output = torch.mul(value[:, 1:], action_mask)
        # return torch.cat((value[:, :1], output), axis=1)

# class DQNModule(nn.Module):
#     def __init__(self, num_states=4, num_actions=18):
#         """
#         Initialize a deep Q-learning network for testing algorithm
#             in_features: number of features of input.
#             num_actions: number of action-value to output, one-to-one correspondence to action in game.
#         """
#         super(DQNModule, self).__init__()
#         self.hidden_size = 128
#         self.fc1 = nn.Linear(num_states, self.hidden_size)
#         self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
#         self.fc3 = nn.Linear(self.hidden_size, num_actions)

#         self.bn1 = nn.BatchNorm1d(self.hidden_size)
#         self.bn2 = nn.BatchNorm1d(self.hidden_size)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         return self.fc3(x)


class DQNTorchPolicy(Policy):
    def __init__(self, observation_space, action_space, config, dqn_config):
        Policy.__init__(self, observation_space, action_space, config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dqn_config = dqn_config
        self.epsilon = 1

        self.num_states =  int(np.product(observation_space.shape))
        self.num_actions = action_space.n
        print(f'dqn state space:{self.num_states}, action space:{self.num_actions}')

        self.eval_net = DQNModule(self.num_states, self.num_actions).to(self.device)
        self.target_net = DQNModule(self.num_states, self.num_actions).to(self.device)

        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter = 0
        self.memory = replay_memory(dqn_config['replay_capacity'])

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=dqn_config['lr'])
        # self.loss_func = nn.SmoothL1Loss()
        self.loss_func = nn.MSELoss()

        self.rand_action = 0
        self.greedy_action = 0

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs): 
        
        if info_batch is None:
            action_dict = [self._action(f_state, None) for f_state in obs_batch ], [], {}  
        else:    
            action_dict = [self._action(f_state, f_state_info) for f_state, f_state_info in zip(obs_batch, info_batch)], [], {}
            
        return action_dict
    
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

    def choose_action(self, state, state_info):
        self.eval()
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.random() >= self.epsilon: # greedy policy
            self.greedy_action += 1
            action_mask = torch.unsqueeze(torch.FloatTensor(state_info['action_mask']), 0).to(self.device)
            with torch.no_grad():
                action_value = self.eval_net(state)
                min_val = torch.min(action_value).item()
                action_value = action_value.masked_fill(action_mask==0, min_val-1.0)
                action = torch.max(action_value, 1)[1].data.cpu().numpy()
                action = action[0]
        else: # random policy
            self.rand_action += 1
            action_mask = state_info['action_mask']
            while True:
                action = np.random.randint(self.num_actions)
                if action_mask[action] == 1:
                    break
            action = action
        return action

    def store_transition(self, state, action, reward, next_state, done):

        state = torch.tensor(state.copy(), dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(float(reward), dtype=torch.float32)
        next_state = torch.tensor(next_state.copy(), dtype=torch.float32)
        done = torch.tensor(float(done), dtype=torch.float32)

        self.memory.push([state, action, reward, next_state, done])


    def learn(self, batch_size):
        if len(self.memory) < self.dqn_config['min_replay_history']:
            return 0, 0

        # update the parameters
        if self.learn_step_counter % self.dqn_config['target_update_period'] == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        self.epsilon = 0.999 * self.epsilon + 0.001 * self.dqn_config['epsilon_train']

        batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.memory.sample(batch_size, device=self.device)

        self.train()

        with torch.no_grad():
            q_next = self.target_net(batch_next_state).detach()
            if self.dqn_config['double_q']:
                q_eval_next = self.eval_net(batch_next_state).detach()
                q_argmax = q_eval_next.max(1)[1]
            else:
                q_argmax = q_next.max(1)[1]
            q_next = q_next.gather(1, q_argmax.unsqueeze(1)).squeeze()
            q_target = batch_reward + self.dqn_config['gamma'] * (1 - batch_done) * q_next

        q_eval = self.eval_net(batch_state).gather(1, batch_action.unsqueeze(1))
        q_eval = q_eval.squeeze()

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del batch_state
        del batch_action
        del batch_reward
        del batch_next_state
        del batch_done
        return loss.item(), q_eval.mean().item()

    def _action(self, state, state_info):
        return self.choose_action(state, state_info)

    def save_param(self, name):
        if not os.path.exists('model'):
            os.mkdir('model')
        torch.save(self.eval_net.state_dict(), f'model/{name}.pkl')

    def load_param(self, name):
        self.eval_net.load_state_dict(torch.load(f'model/{name}.pkl'))
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def train(self):
        self.eval_net.train()
        self.target_net.train()

    def eval(self):
        self.eval_net.eval()
        self.target_net.eval()



class Trainer():
    def __init__(self, env, policy, config):
        self.env = env
        self.step = 0
        self.policy = policy
        self.training_timestep = config['training_steps']
        self.batch_size = config['batch_size']
        self.update_period = config['update_period']

    def save(self, name, iter):
        self.policy.save_param(name + f"_iter_{iter}")

    def get_policy(self):
        return self.policy

    def train(self, iter):
        print(f"  == iteration {iter} == ")
        rewards_all = []
        episode_reward_all = []
        train_loss_list = []
        qvalue_list = []
        episode_reward = 0.0

        state, infos = self.env.reset()
        action_distribution = list(range(self.policy.num_actions))

        for i in range(self.training_timestep):
            self.step += 1
            # print("timestep : ", self.step)
            # print("Start calculate action ....")
            action, _, _ = self.policy.compute_single_action( state, info=infos, explore=True )
            next_state, reward, done, infos = self.env.step(action)
           
            rewards_all.append(reward)
            episode_reward += reward

            self.policy.store_transition(state, action, reward, next_state, done)
            action_distribution[action] += 1

            if self.step % self.update_period == 0:
                loss, qvalue = self.policy.learn(self.batch_size)
                train_loss_list.append(loss)
                qvalue_list.append(qvalue)
            if done:
                state, infos = self.env.reset()
                episode_reward_all.append(episode_reward)
                episode_reward = 0.0
            else:
                state = next_state
        return {
            "episode_step": self.training_timestep,
            "rewards_all": rewards_all,
            "episode_reward_all": episode_reward_all,
            "train_loss": train_loss_list,
            "train_qvalue": qvalue_list,
            "action_distribution": action_distribution,
            "epsilon": self.policy.epsilon
        }
