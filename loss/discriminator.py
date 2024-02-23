import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_channels=1, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv3d(input_channels, features[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, feature, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(feature),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = feature

        self.model = nn.Sequential(*layers)
        # Adjust final conv layer to match the reduced size or add adaptive pooling if necessary
        self.final = nn.Conv3d(in_channels, 1, kernel_size=(1,4,4), stride=1, padding=0)

    def forward(self, x):
        x = x[None,...]
        x = self.initial(x)
        for layer in self.model:
            x = layer(x)
        x = self.final(x)
        # Flatten the output for compatibility with loss functions
        return torch.sigmoid(x.view(x.size(0), -1))

# Check the model
if __name__ == '__main__':
    discriminator = Discriminator()
    print(discriminator)

    # Sample input tensor of size [batch_size, channels, depth, height, width]
    input_tensor = torch.randn(1, 1, 16, 64, 64)
    output = discriminator(input_tensor)
    print(f"Output size: {output.size()}")