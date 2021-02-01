import global_config
import numpy as np

class Trainer():
    def __init__(self, env, policies, policies_to_train, config):
        self.env = env
        self.step = 0
        self.policies = policies
        self.policies_to_train = policies_to_train
        self.training_steps = config['training_steps']
        # self.eval_steps = config['eval_steps']
        self.batch_size = config['batch_size']
        self.update_period = config['update_period']
       
    def save(self, name, iter):
        for agent_id in self.policies_to_train:
            policy = self.policies[agent_id]
            policy.save_param(name + f"_{agent_id}_iter_{iter}")

    def restore(self, name, iter):
        for agent_id in self.policies_to_train:
            policy = self.policies[agent_id]
            policy.load_param(name + f"_{agent_id}_iter_{iter}")

    def switch_mode(self, eval):  # only for epsilon greedy
        for policy in self.policies_to_train:
            self.policies[policy].switch_mode(eval=eval)

    def get_policy(self, agent_id):
        return self.policies[agent_id]

    def train(self, iter):
        self.switch_mode(eval=False)
        print(f"  == iteration {iter} == ")

        self.env.set_training_mode(True)
        obss = self.env.reset()
        _, infos = self.env.state_calculator.world_to_state(self.env.world)
        rnn_states = {}
        rewards_all = {}
        episode_reward_all = {}
        episode_reward = {}
        policies_to_train_loss = {key : [] for key in self.policies_to_train}
        policies_to_train_qvalue = {key : [] for key in self.policies_to_train}
        if self.env.is_reward_shape_on:
            action_distribution = {key : [0 for i in range(int(np.product(self.env.action_space_consumer.shape)))] for key in self.policies_to_train}
            for key in self.policies_to_train:
                action_distribution[key] = {}
                for i in range(self.env.action_space_consumer.nvec[0]):
                    for j in range(self.env.action_space_consumer.nvec[1]):
                        action_distribution[key][(i,j)] = 0
        else:
            action_distribution = {key : [0 for i in range(int(self.env.action_space_consumer.n))] for key in self.policies_to_train}
        episode_steps = []
        episode_step = 0

        for agent_id in obss.keys():
            # policies[agent_id] = load_policy(agent_id)
            rnn_states[agent_id] = self.policies[agent_id].get_initial_state()
            rewards_all[agent_id] = []
            episode_reward_all[agent_id] = []
            episode_reward[agent_id] = 0

        for i in range(self.training_steps):
            self.step += 1
            episode_step += 1
            actions = {}
            actions_train = {}
            # print("timestep : ", self.step)
            # print("Start calculate action ....")
            for agent_id, obs in obss.items():
                policy = self.policies[agent_id]
                action, new_state, _ = policy.compute_single_action(obs, state=rnn_states[agent_id],
                                                                    info=infos[agent_id],
                                                                    explore=True)
                if self.env.is_reward_shape_on and agent_id in self.policies_to_train:
                    actions[agent_id] = self.policies[agent_id].policy_action_space[action]
                else:
                    actions[agent_id] = action
                actions_train[agent_id] = action

                # print(agent_id, " :", policy.__class__, " : ", action)
            next_obss, rewards, dones, infos = self.env.step(actions)

            for agent_id, reward in rewards.items():
                rewards_all[agent_id].append(reward)
                episode_reward[agent_id] += reward

            done = any(dones.values())

            for agent_id in self.policies_to_train:
                self.policies[agent_id].store_transition(obss[agent_id],
                                                         actions_train[agent_id],
                                                         rewards[agent_id],
                                                         next_obss[agent_id],
                                                         done)
                action_distribution[agent_id][actions[agent_id]] += 1

            # if self.step % (self.update_period * len(self.policies_to_train)) == 0:
            if self.step % self.update_period == 0:
                for agent_id in self.policies_to_train:
                    loss, qvalue = self.policies[agent_id].learn(self.batch_size)
                    policies_to_train_loss[agent_id].append(loss)
                    policies_to_train_qvalue[agent_id].append(qvalue)
            if done:
                obss = self.env.reset()
                episode_steps.append(episode_step)
                episode_step = 0
                for agent_id, reward in episode_reward.items():
                    episode_reward_all[agent_id].append(reward)
                    episode_reward[agent_id] = 0
            else:
                obss = next_obss
        infos = {
            "rewards_all": rewards_all,
            "episode_reward_all": episode_reward_all,
            "policies_to_train_loss": policies_to_train_loss,
            "policies_to_train_qvalue": policies_to_train_qvalue,
            "action_distribution": action_distribution,
            "epsilon": self.policies[self.policies_to_train[0]].epsilon,
            "all_step": self.step,
            "episode_step": sum(episode_steps) / len(episode_steps),
        }
        # dqn_policy = self.policies[self.policies_to_train[0]]
        # print('random action: ', dqn_policy.rand_action, ' greedy action:', dqn_policy.greedy_action)
        # dqn_policy.rand_action = 0
        # dqn_policy.greedy_action = 0
        return infos

    def eval(self, iter):
        self.switch_mode(eval=True)
        print(f"  == eval iteration {iter} == ")
        self.env.set_training_mode(False)
        obss = self.env.reset()
        _, infos = self.env.state_calculator.world_to_state(self.env.world)
        rnn_states = {}
        rewards_all = {}
        episode_reward_all = {}
        episode_reward = {}
        episode_steps = []
        episode_step = 0

        for agent_id in obss.keys():
            # policies[agent_id] = load_policy(agent_id)
            rnn_states[agent_id] = self.policies[agent_id].get_initial_state()
            rewards_all[agent_id] = []
            episode_reward_all[agent_id] = []
            episode_reward[agent_id] = 0

        for i in range(100000):
            episode_step += 1
            actions = {}
            # print("timestep : ", self.step)
            # print("Start calculate action ....")
            for agent_id, obs in obss.items():
                policy = self.policies[agent_id]
                action, new_state, _ = policy.compute_single_action(obs, state=rnn_states[agent_id],
                                                                    info=infos[agent_id],
                                                                    explore=False)
                if self.env.is_reward_shape_on and agent_id in self.policies_to_train:
                    actions[agent_id] = self.policies[agent_id].policy_action_space[action]
                else:
                    actions[agent_id] = action
                # print(agent_id, " :", policy.__class__, " : ", action)
            next_obss, rewards, dones, infos = self.env.step(actions)

            for agent_id, reward in rewards.items():
                rewards_all[agent_id].append(reward)
                episode_reward[agent_id] += reward

            done = any(dones.values())

            if done:
                obss = self.env.reset()
                episode_steps.append(episode_step)
                episode_step = 0
                for agent_id, reward in episode_reward.items():
                    episode_reward_all[agent_id].append(reward)
                    episode_reward[agent_id] = 0
                break
            else:
                obss = next_obss
        infos = {
            "rewards_all": rewards_all,
            "episode_reward_all": episode_reward_all,
            "epsilon": self.policies[self.policies_to_train[0]].epsilon,
            "all_step": self.step,
            "episode_step": sum(episode_steps) / len(episode_steps),
        }
        return infos

    def load_data(self, eval=False):
        # self.switch_mode(eval=False)
        print(f"  == start load data eval={eval} == ")

        obss = self.env.reset(eval=eval)
        _, infos = self.env.state_calculator.world_to_state(self.env.world)
        episode_step = 0

        uncontrollable_part_state = {key: infos[key][self.uncontrollable_part_state_key].copy() for key in self.policies_to_train}
        uncontrollable_part_pred = {key: infos[key]['uncontrollable_part_pred'].copy() for key in self.policies_to_train}


        for i in range(self.training_steps):
            episode_step += 1
            actions = {}
            for agent_id, obs in obss.items():
                policy = self.policies[agent_id]
                action, new_state, _ = policy.compute_single_action(obs, state=None,
                                                                    info=infos[agent_id],
                                                                    explore=True)
                actions[agent_id] = action
            next_obss, rewards, dones, infos = self.env.step(actions)

            done = any(dones.values())

            for agent_id in self.policies_to_train:
                # print('policies_to_train: ', agent_id, " reward: ", rewards[agent_id])
                self.policies[agent_id].store_transition(obss[agent_id],
                                                         actions[agent_id],
                                                         rewards[agent_id],
                                                         next_obss[agent_id],
                                                         done,
                                                         uncontrollable_part_state[agent_id],
                                                         uncontrollable_part_pred[agent_id],
                                                         infos[agent_id][self.uncontrollable_part_state_key],
                                                         agent_id, eval=eval)
                uncontrollable_part_state[agent_id] = infos[agent_id][self.uncontrollable_part_state_key].copy()
                uncontrollable_part_pred[agent_id] = infos[agent_id]['uncontrollable_part_pred'].copy()

            if done:
                break
            else:
                obss = next_obss
        print(f"  == load data end episode len={episode_step}==")

