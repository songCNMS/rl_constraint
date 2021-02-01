from scheduler.inventory_random_policy import BaselinePolicy, ConsumerBaselinePolicy
from ray.rllib.models.torch.misc import SlimFC, normc_initializer as \
    torch_normc_initializer

from utility.replay_memory import replay_memory
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os

class DQNModule(nn.Module):
    def __init__(self, num_states=4, num_actions=18, pred=False):
        """
        Initialize a deep Q-learning network for testing algorithm
            in_features: number of features of input.
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQNModule, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.hidden_size = 128

        # self.fc1 = nn.Linear(num_states, self.hidden_size)
        # self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc3 = nn.Linear(self.hidden_size, num_actions)
        prev_layer_size = num_states
        self.hidden_out_size = num_actions
        layers = []
        for size in [128, 32, self.hidden_out_size]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=torch_normc_initializer(1.0),
                    activation_fn=nn.ReLU))
            prev_layer_size = size
            # Add a batch norm layer.
            layers.append(nn.BatchNorm1d(prev_layer_size))
        self._hidden_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)
        return self._hidden_layers(x)


class ConsumerDQNTorchPolicy(BaselinePolicy):

    def __init__(self, observation_space, action_space, config, dqn_config):
        BaselinePolicy.__init__(self, observation_space, action_space, config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dqn_config = dqn_config
        self.epsilon = 1

        self.num_states =  int(np.product(observation_space.shape))
        self.num_actions = int(action_space.n)
        print(f'dqn state space:{self.num_states}, action space:{self.num_actions}')
        self.pred_head = False

        self.eval_net = DQNModule(self.num_states, self.num_actions, self.pred_head).to(self.device)
        self.target_net = DQNModule(self.num_states, self.num_actions, self.pred_head).to(self.device)

        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter = 0
        self.memory = replay_memory(dqn_config['replay_capacity'])

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=dqn_config['lr'])
        self.loss_func = nn.SmoothL1Loss()

        self.rand_action = 0
        self.greedy_action = 0
        self.evaluation = False

    def switch_mode(self, eval=False): # only for epsilon-greedy
        self.evaluation = eval

    def set_weights(self, net_parameter_dict):
        self.eval_net.load_state_dict(net_parameter_dict['eval_net'])
        self.target_net.load_state_dict(net_parameter_dict['target_net'])
        print('load weights')

    def get_weights(self):
        return {'eval_net': self.eval_net.state_dict(), 'target_net': self.target_net.state_dict()}

    def choose_action(self, state, infos):

        self.eval()
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.random() >= self.epsilon or self.evaluation: # greedy policy
            self.greedy_action += 1
            with torch.no_grad():
                action_value = self.eval_net(state)
                action = torch.max(action_value, 1)[1].data.cpu().numpy()
                action = action[0]
        else: # random policy
            self.rand_action += 1
            action = np.random.randint(self.num_actions)
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
            q_next = self.target_net(batch_next_state)
            q_next = q_next.detach()
            if self.dqn_config['double_q']:
                q_eval_next = self.eval_net(batch_next_state)
                q_eval_next = q_eval_next.detach()
                q_argmax = q_eval_next.max(1)[1]
            else:
                q_argmax = q_next.max(1)[1]
            q_next = q_next.gather(1, q_argmax.unsqueeze(1)).squeeze()
            q_target = batch_reward + self.dqn_config['gamma'] * (1 - batch_done) * q_next

        q_eval = self.eval_net(batch_state)
        q_eval = q_eval.gather(1, batch_action.unsqueeze(1))
        q_eval = q_eval.squeeze()

        loss = self.loss_func(q_eval, q_target)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), q_eval.mean().item()


    def _action(self, state, state_info):
        return self.choose_action(state, state_info)

    def save_param(self, name):
        if not os.path.exists('model'):
            os.mkdir('model')
        torch.save(self.eval_net.state_dict(), f'model/{name}.pkl')

    def load_param(self, name):
        self.eval_net.load_state_dict(torch.load(f'model/{name}.pkl'))
        self.target_net.load_state_dict(torch.load(f'model/{name}.pkl'))

    def train(self):
        self.eval_net.train()
        self.target_net.train()

    def eval(self):
        self.eval_net.eval()
        self.target_net.eval()



