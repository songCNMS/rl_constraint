from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch
from ray.rllib.models.torch.misc import SlimFC, normc_initializer as \
    torch_normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import numpy as np
from gym.spaces import Discrete

torch, nn = try_import_torch()


class VRPModel(TorchModelV2, nn.Module):
    """Example of a TorchModelV2 using batch normalization."""
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        self.hidden_out_size = 64
        for size in [512, 128, self.hidden_out_size]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=torch_normc_initializer(1.0),
                    activation_fn=nn.ReLU))
            prev_layer_size = size
            # Add a batch norm layer.
            # layers.append(nn.BatchNorm1d(prev_layer_size))

        self._logits = SlimFC(
            in_size=prev_layer_size,
            out_size=num_outputs,
            initializer=torch_normc_initializer(0.1),
            activation_fn=None)

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=torch_normc_initializer(0.1),
            activation_fn=None)

        self._hidden_layers = nn.Sequential(*layers)
        self._hidden_out = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Set the correct train-mode for our hidden module (only important
        # b/c we have some batch-norm layers).
        self._hidden_layers.train(mode=input_dict.get("is_training", False))
        self._hidden_out = self._hidden_layers(input_dict["obs"])
        logits = self._logits(self._hidden_out)
        logits = torch.clamp(logits, -10.0, 10.0)
        return logits, []

    @override(ModelV2)
    def value_function(self):
        assert self._hidden_out is not None, "must call forward first!"
        return torch.reshape(self._value_branch(self._hidden_out), [-1])



class KnapsackModel(TorchModelV2, nn.Module):
    """Example of a TorchModelV2 using batch normalization."""
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        self.hidden_out_size = 8
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

        self._logits = SlimFC(
            in_size=prev_layer_size,
            out_size=num_outputs,
            initializer=torch_normc_initializer(0.1),
            activation_fn=None)

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=torch_normc_initializer(0.1),
            activation_fn=None)

        self._hidden_layers = nn.Sequential(*layers)
        self._hidden_out = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Set the correct train-mode for our hidden module (only important
        # b/c we have some batch-norm layers).
        self._hidden_layers.train(mode=input_dict.get("is_training", False))
        self._hidden_out = self._hidden_layers(input_dict["obs"])
        logits = self._logits(self._hidden_out)
        logits = torch.clamp(logits, -10.0, 10.0)
        return logits, []

    @override(ModelV2)
    def value_function(self):
        assert self._hidden_out is not None, "must call forward first!"
        return torch.reshape(self._value_branch(self._hidden_out), [-1])