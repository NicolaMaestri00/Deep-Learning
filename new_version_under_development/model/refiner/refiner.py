#####################################
# Refiner
#####################################

import torch
import torch.nn as nn


class Refiner(nn.Module):
    """  Refiner: upsamples the low level probability map to a pixel probability map    """

    def __init__(self, patch_per_image=14) -> None:
        super(Refiner, self).__init__()

        self.patch_per_image = patch_per_image

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1+768, out_channels=1, kernel_size=3, padding='same')
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(1)

        self.conv2 = nn.Conv2d(in_channels=1+768, out_channels=1, kernel_size=3, padding='same')
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(1)

        self.conv3 = nn.Conv2d(in_channels=1+768, out_channels=1, kernel_size=3, padding='same')
        self.relu3 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm2d(1)

        self.conv4 = nn.Conv2d(in_channels=1+768, out_channels=5, kernel_size=3, padding='same')
        self.relu4 = nn.ReLU()
        self.batchnorm4 = nn.BatchNorm2d(5)

        # Fully connected layers
        self.conv5 = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=5, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, maps: torch.Tensor, fv) -> torch.Tensor:
        """
        Forward pass 

        Args:
        maps:  low level probability map (batch x 14 x 14)
        fv: image tokens encoded at different layers of the vision transformer [fv1, fv2, fv3, fv4]
        fv1:  image tokens encoded at layer 1   (batch x (1+n_patches) x 768)
        fv2:  image tokens encoded at layer 2   (batch x (1+n_patches) x 768)
        fv3:  image tokens encoded at layer 3   (batch x (1+n_patches) x 768)
        fv4:  image tokens encoded at layer 4   (batch x (1+n_patches) x 768)
        
        Returns:
        x:  pixel probability map (batch x 224 x 224)
        """
        # Remove cls token and reshape
        fv1 = fv[0][:, 1:, :].view(-1, 768, 14, 14)
        fv2 = fv[1][:, 1:, :].view(-1, 768, 14, 14)
        fv3 = fv[2][:, 1:, :].view(-1, 768, 14, 14)
        fv4 = fv[3][:, 1:, :].view(-1, 768, 14, 14)

        # First upsample    14x14 -> 28x28
        x = maps
        x = torch.cat((x, fv4), dim=1)          # Concatenate low level probability map with image tokens
        x = self.conv1(x)                           
        x = self.relu1(x)
        x = self.batchnorm1(x)
        x = x + maps                                                                # Residual connection                                                          
        x = nn.functional.interpolate(input=x, mode='bilinear', scale_factor=2)     # Upsample

        # Second upsample    28x28 -> 56x56
        y = x
        fv3 = nn.functional.interpolate(input=fv3, mode='bilinear', scale_factor=2)
        x = torch.cat((x, fv3), dim=1)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)
        x = x + y
        x = nn.functional.interpolate(input=x, mode='bilinear', scale_factor=2)

        # Third upsample    56x56 -> 112x112
        y = x
        fv2 = nn.functional.interpolate(input=fv2, mode='bilinear', scale_factor=4)
        x = torch.cat((x, fv2), dim=1)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.batchnorm3(x)
        x = x + y
        x = nn.functional.interpolate(input=x, mode='bilinear', scale_factor=2)

        # Fourth upsample    112x112 -> 224x224
        y = x
        fv1 = nn.functional.interpolate(input=fv1, mode='bilinear', scale_factor=8)
        x = torch.cat((x, fv1), dim=1)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.batchnorm4(x)
        x = x + y.repeat(1, 5, 1, 1)
        x = nn.functional.interpolate(input=x, mode='bilinear', scale_factor=2)

        # Sigmoid
        x = self.conv5(x)
        x = x.view(-1, 1, 224*224)
        x = self.sigmoid(x)
        x = x.view(-1, 1, 224, 224)
        x = x.squeeze(1)

        return x