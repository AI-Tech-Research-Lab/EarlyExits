# add file path to sys.path
import copy
import os
import sys
# Get the home directory
home_dir = os.path.expanduser('~')  # This will return the home directory path (e.g., /home/username)
# Concatenate the home directory with the 'EarlyExits/models' path
model_dir = os.path.join(home_dir, 'workspace/CNAS/EarlyExits', 'models')
sys.path.append(model_dir)
model_dir = os.path.join(home_dir, 'workspace/CNAS/EarlyExits')
sys.path.append(model_dir)

from EarlyExits.models.base import BranchModel
import torch
import torch.nn as nn
import torchvision.models as models
'''
from base import BranchModel, IntermediateBranch
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from utils_ee import get_intermediate_backbone_cost, get_intermediate_classifiers_adaptive, get_intermediate_classifiers_cost, get_intermediate_classifiers_static
'''

def replace_silu_with_relu(model):
    """
    Recursively traverse the model and replace all SiLU activations with
    two ReLU activations.
    """
    for child_name, child in model.named_children():
        # If the child is SiLU, replace it with two ReLU
        if isinstance(child, nn.SiLU):
            relu1 = nn.ReLU(inplace=True)
            relu2 = nn.ReLU(inplace=True)
            # Substitute the SiLU with a Sequential containing two ReLU activations
            setattr(model, child_name, nn.Sequential(relu1, relu2))
        
        # Recursively apply to submodules
        replace_silu_with_relu(child)

class EfficientNetClassifier(nn.Module):
    def __init__(self, n_classes):
        # Initialize the sequential layers
        super(EfficientNetClassifier, self).__init__()
        self.seq = nn.AdaptiveAvgPool2d(1)  # Pooling to 1x1 size
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),  # Dropout for regularization
            nn.Linear(1280, n_classes, bias=True)  # Fully connected layer
        )
    def forward(self, x):
        x = self.seq(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class EEEfficientNet(BranchModel, nn.Module):

    def __init__(self, backbone):
        super(EEEfficientNet, self).__init__()

        self.efficientnet = backbone #models.efficientnet_b0(weights=None)
        self.b = 4 #sum(branches) + 1  # +1 for final exit
        self.first_conv = self.efficientnet.features[0]
        self.blocks = self.efficientnet.features[1:]
        print(f"Number of layers: {len(self.blocks)}")
        self.exit_idxs = [0, 2, 4, 7]
        
    def n_branches(self):
        return self.b

    def forward(self, x):
        i = 0
        intermediate_layers = []
        x = self.first_conv(x)

        # Iterate over the blocks
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx == self.exit_idxs[i]:  # Early exit point
                #intermediate_output = self.predictors[i](x)
                intermediate_layers.append(x)
                i += 1
        
        return intermediate_layers

'''
n_classes=10
img_size=(3,160,160)
backbone=EEEfficientNet()
# Generate intermediate classifiers for the early exit points
final_classifier = EfficientNetClassifier(n_classes)
net = copy.deepcopy(backbone)
replace_silu_with_relu(net)
predictors = get_intermediate_classifiers_adaptive(net, final_classifier, img_size, n_classes=n_classes, model_name='efficientnet')

b_params, b_macs = get_intermediate_backbone_cost(net, img_size)
print(b_params, b_macs)
c_params, c_macs = get_intermediate_classifiers_cost(backbone, predictors, img_size)
print(c_params, c_macs)
'''




