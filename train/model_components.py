import torch as th
import torch.nn as nn
import gymnasium.spaces as spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MixedFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.image_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 4)),
            nn.Flatten(),
        )
        dummy_image = th.zeros(1, *observation_space["st_image"].shape)
        cnn_output_dim = self.image_net(dummy_image).shape[1]
        self.array_net = nn.Sequential(
            nn.Linear(observation_space["array"].shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> th.Tensor:
        image_features = self.image_net(observations["st_image"])
        array_features = self.array_net(observations["array"])

        combined_features = th.cat([image_features, array_features], dim=1)
        return self.fc(combined_features)

