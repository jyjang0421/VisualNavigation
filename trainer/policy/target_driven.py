import torchvision.models as model
import torch
import torchvision
from torch import nn as nn

import numpy as np

from typing import Dict

class ResnetEncoder(nn.Module):
    def __init__(
        self,
        output_size,
    ):
        super().__init__()
        #resnet fc layer 제거
        self.resnet = self._load_resnet50(True)
        self.fc1 = nn.Linear(2048, 512, bias=True)
        self.fc2 = nn.Linear(1024, output_size, bias=True)
        
    def _load_resnet50(self, pretrained):
        resnet = model.resnet50( pretrained = pretrained, progress = False )
        #fully_connected layer 제거
        resnet = torch.nn.Sequential( *list(resnet.children())[:-1] )
        return resnet
    
    def forward(self, observations: Dict[str, torch.Tensor]):
        current_obs = observations["current_rgb"]
        target_obs = observations["target_rgb"]
        
        #2048d로 변경
        def change_dim(x):
            x.squeeze_(-1)
            x.squeeze_(-1)
            x_ = x.view([1,-1])
            x_.squeeze_(0)
            return x_
        
        #observation & target input tuple을 tensor로 변경
        def change_input_dim(x):
            x = torch.from_numpy(x)
            x = x.reshape(-1,256,256)
            return x
        
        #####resnet_layers
        obs = change_input_dim(current_obs)
        #2048d로 만들기
        obs = obs.unsqueeze(0)
        obs_= obs.float()
        #resnet통과 & fcLayer input차원으로 바꾸기
        obs = self.resnet(obs_)
        obs = change_dim(obs)
        
        t = change_input_dim(target_obs)
        t = t.unsqueeze(0)
        t = self.resnet(t)
        t = change_dim(t)
        
        #####fc_layers
        ob_feature = self.fc1(obs)
        tg_feature = self.fc1(t)
        
        #512+512 = 1024d로 만들기
        output = torch.cat([ob_feature,tg_feature])
        
        #####embedding fusion
        result = self.fc2(output)
        return result