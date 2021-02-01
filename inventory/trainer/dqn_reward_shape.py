from ray.rllib.utils import try_import_torch
from ray.rllib.models.torch.misc import SlimFC, normc_initializer as \
    torch_normc_initializer

from scheduler.inventory_random_policy import BaselinePolicy, ConsumerBaselinePolicy
from utility.replay_memory import replay_memory
import torch.nn as nn
import torch.nn.functional as F
import torch
import os

import numpy as np
from gym.spaces import Discrete

torch, nn = try_import_torch()


# output reward discount value for ConsumerUnit
class ConsumerRewardDiscountModel():

    def __init__(self, num_states, num_actions, device):
        layers = []
        prev_layer_size = num_states

        # Create layers 0 to second-last.
        self.hidden_out_size = 8
        for size in [32, self.hidden_out_size]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=torch_normc_initializer(1.0),
                    activation_fn=nn.ReLU))
            prev_layer_size = size

        self._hidden_layers = nn.Sequential(*layers).to(device)
        self._hidden_out = None
        self.num_outputs = num_actions


class SKUStoreBatchNormModel():

    def __init__(self, num_states, num_actions, device):
        layers = []
        prev_layer_size = num_states
        self._logits = None

        # Create layers 0 to second-last.
        self.hidden_out_size = 16
        for size in [64, 32, self.hidden_out_size]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=torch_normc_initializer(1.0),
                    activation_fn=nn.ReLU))
            prev_layer_size = size
            # Add a batch norm layer.
            # layers.append(nn.BatchNorm1d(prev_layer_size))
        self._hidden_layers = nn.Sequential(*layers).to(device)
        self._hidden_out = None
        self.num_outputs = num_actions


class ConsumerRewardShapeModel(nn.Module):
    def __init__(self, num_states, num_actions):
        nn.Module.__init__(self)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.reward_discount_model = ConsumerRewardDiscountModel(num_states, num_actions[0], self.device)
        self.consumer_model = SKUStoreBatchNormModel(num_states, num_actions[1], self.device)
        self.consumer_logits = SlimFC(in_size=self.consumer_model.hidden_out_size + self.consumer_model.num_outputs,
                              out_size=self.consumer_model.num_outputs,
                              initializer=torch_normc_initializer(0.1),
                              activation_fn=None)
        
        self.explicit_reward_discount_for_consumer = SlimFC(in_size=self.reward_discount_model.hidden_out_size,
                                                            out_size=self.consumer_model.num_outputs,
                                                            initializer=torch_normc_initializer(0.1),
                                                            activation_fn=None)
        
        self.reward_discount_logits = SlimFC(2*self.consumer_model.num_outputs,
                                            out_size=self.reward_discount_model.num_outputs,
                                            initializer=torch_normc_initializer(0.1),
                                            activation_fn=None)

        self.discount_training = True
        self._hidden_out = None
        self.output_layer = SlimFC(self.consumer_model.num_outputs + self.reward_discount_model.num_outputs,
                                    out_size=num_actions[0]*num_actions[1],
                                    initializer=torch_normc_initializer(0.1),
                                    activation_fn=None)

    def forward(self, x):
        x = x.to(self.device)
        for model in [self.consumer_model._hidden_layers, self.consumer_logits,
                      self.reward_discount_model._hidden_layers, self.reward_discount_logits, 
                      self.explicit_reward_discount_for_consumer]:
            for p in model.parameters():
                p.requires_grad = True

        if self.discount_training:
            for model in [self.consumer_model._hidden_layers, self.consumer_logits]:
                for p in model.parameters():
                    p.requires_grad = False
        else:
            for model in [self.reward_discount_model._hidden_layers, 
                          self.reward_discount_logits,
                          self.explicit_reward_discount_for_consumer]:
                for p in model.parameters():
                    p.requires_grad = False
        self.reward_discount_model._hidden_out = self.reward_discount_model._hidden_layers(x)
        explicit_reward_discount_for_consumer_out = self.explicit_reward_discount_for_consumer(self.reward_discount_model._hidden_out)
        
        self.consumer_model._hidden_out = self.consumer_model._hidden_layers(x)
        self._hidden_out = torch.cat((self.consumer_model._hidden_out, explicit_reward_discount_for_consumer_out), axis=1)

        consumer_logits_out = self.consumer_logits(self._hidden_out)
        reward_logits_input = torch.cat((explicit_reward_discount_for_consumer_out, consumer_logits_out), axis=1)
        reward_discount_logits_out = self.reward_discount_logits(reward_logits_input)
        logits = torch.cat((reward_discount_logits_out, consumer_logits_out), axis=1)
        return self.output_layer(logits)


class ConsumerRewardShapeDQNTorchPolicy(BaselinePolicy):

    def __init__(self, observation_space, action_space, config, dqn_config):
        BaselinePolicy.__init__(self, observation_space, action_space, config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dqn_config = dqn_config
        self.epsilon = 1

        self.num_states =  int(np.product(observation_space.shape))
        self.num_actions = action_space.nvec

        self.policy_action_space = []
        for i in range(self.num_actions[0]):
            for j in range(self.num_actions[1]):
                self.policy_action_space.append((i, j))

        print(f'dqn state space:{self.num_states}, action space:{self.num_actions}')
        self.pred_head = False

        self.eval_net = ConsumerRewardShapeModel(self.num_states, self.num_actions).to(self.device)
        self.target_net = ConsumerRewardShapeModel(self.num_states, self.num_actions).to(self.device)

        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter = 0
        self.discount_training = True
        self.memory = replay_memory(dqn_config['replay_capacity'])
        self.discount_memory = replay_memory(dqn_config['replay_capacity'])

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
            action = np.random.randint(len(self.policy_action_space))
        return action

    def store_transition(self, state, action, reward, next_state, done):

        state = torch.tensor(state.copy(), dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(float(reward), dtype=torch.float32)
        next_state = torch.tensor(next_state.copy(), dtype=torch.float32)
        done = torch.tensor(float(done), dtype=torch.float32)
        if self.discount_training:
            self.discount_memory.push([state, action, reward, next_state, done])
        else:
            self.memory.push([state, action, reward, next_state, done])


    def learn(self, batch_size):
        if self.discount_training and len(self.discount_memory) < self.dqn_config['min_replay_history']:
            return 0, 0
        if ~self.discount_training and len(self.memory) < self.dqn_config['min_replay_history']:
            return 0, 0

        # update the parameters
        if self.learn_step_counter % self.dqn_config['target_update_period'] == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        self.epsilon = 0.999 * self.epsilon + 0.001 * self.dqn_config['epsilon_train']

        if self.discount_training:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.discount_memory.sample(batch_size, device=self.device)
        else:
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
        self.policy.load_state_dict(torch.load(f'model/{name}.pkl'))

    def train(self):
        self.eval_net.train()
        self.target_net.train()

    def eval(self):
        self.eval_net.eval()
        self.target_net.eval()