from baseline import BaselineImgConv2d, BaselineTopoConv1d, BaselineMlpClassifier
from cnntdanet import CNNTDAPlus
from utils import compute_output_dim

import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self,
                 image_shape,
                 target_shape):
        super(BaselineCNN, self).__init__()
        self.feature_extractor = BaselineImgConv2d(image_shape)
        self.classifier = BaselineMlpClassifier(compute_output_dim(image_shape),target_shape)

    def forward(self, x):
        feature_map = self.feature_extractor(x)
        output = self.classifier(feature_map)
        return output

    @property
    def module(self):
        """
        If the model is being used with DataParallel, 
        this property will return the actual model.
        """
        return self._modules.get('module', self)



    