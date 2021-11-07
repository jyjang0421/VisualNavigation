#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import torch
from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict
from torch import nn as nn

from habitat.config import Config
from habitat.tasks.nav.nav import (
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Policy, PointNavBaselinePolicy
from trainer.policy.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from habitat_baselines.utils.common import CategoricalNet
from trainer.policy.target_driven import ResnetEncoder


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)
    
@baseline_registry.register_policy(name="TargetDrivenPolicy")
class TargetDrivenPolicy(Policy):
    def __init__(
        self,
        observation_space: SpaceDict,
        action_space,
        hidden_size: int = 512,
        **kwargs
    ):
        super().__init__(
            TargetDrivenNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                **kwargs,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: SpaceDict, action_space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class TargetDrivenNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: SpaceDict,
        hidden_size: int,
    ):
        super().__init__()

        self._hidden_size = hidden_size

        self.visual_encoder = ResnetEncoder(hidden_size)

        self.state_encoder = RNNStateEncoder(
            self._hidden_size,
            self._hidden_size,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        obs = {}
        
        if ImageGoalSensor.cls_uuid in observations:
            image_goal = observations[ImageGoalSensor.cls_uuid]
            obs["target_rgb"] = image_goal
            
        obs["current_rgb"] = observations["rgb"]
        
        x = self.visul_encoder(obs)
        
        
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states