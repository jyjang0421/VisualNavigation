#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Tuple

import numpy as np
import torch
from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict
from torch import nn as nn
from torch.nn import functional as F

from habitat.config import Config
from habitat.tasks.nav.nav import ImageGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
#내 resnetEncoder 사용하기 위해 import
from trainer.policy.target_driven import ResnetEncoder
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo import Net, Policy
from habitat_baselines.utils.common import Flatten


@baseline_registry.register_policy
class PointNavResNetPolicy(Policy):
    def __init__(
        self,
        observation_space: SpaceDict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 2,
        rnn_type: str = "LSTM",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet50",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        **kwargs
    ):
        super().__init__(
            PointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                force_blind_policy=force_blind_policy,
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
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            backbone=config.RL.DDPPO.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.FORCE_BLIND_POLICY,
        )



    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.is_blind:
            return None

        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)

            cnn_input.append(depth_observations)

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class PointNavResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: SpaceDict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        force_blind_policy: bool = False,
    ):
        super().__init__()

        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action
             
        self._hidden_size = hidden_size
        #visual_encoder 내껄로 쓰기.
        self.visual_encoder =ResnetEncoder(hidden_size)


        self.state_encoder = RNNStateEncoder(
            self._hidden_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> Tuple[torch.Tensor]:

        if ImageGoalSensor.cls_uuid in observations:
            image_goal = observations[ImageGoalSensor.cls_uuid]
            obs["target_rgb"] = image_goal
            obs["current_rgb"] = observations["rgb"]
            
        if PanoramicImageGoalSensor.cls_uuid in observations:
            image_goal = observations[PanoramicImageGoalSensor.cls_uuid]
            obs["target_rgb"] = image_goal
            obs["current_rgb"] = observations["panoramic_rgb"]
            
        if PanoramicImageGoalSensor.cls_uuid in observations:
            image_goal = observations["cube_image_goal"]
            obs["target_rgb"] = image_goal
            obs["current_rgb"] = observations["cube_rgb"]
              
        x = self.visual_encoder(obs)
        
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states
