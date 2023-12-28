import torch
import torch.nn as nn

# Model in research paper
class cnn_105_encoder(nn.Module):

    def __init__(self):
        super().__init__()

        image_size = (105, 105, 3)
        num_channels = image_size[-1]

        self.block1 = self.get_cnn_block(num_channels, 64, 10)
        self.block2 = self.get_cnn_block(64, 128, 7)
        self.block3 = self.get_cnn_block(128, 128, 4)
        self.block4 = self.get_cnn_block(128, 256, 4, last_block=True)
        self.flat = nn.Flatten(start_dim=1, end_dim=-1) # start_dim=1 Do not flatten batch fim
        self.fc1 = nn.Linear(9216, 4096)
        self.act = nn.Sigmoid()

    def forward(self, x1):
        x2 = self.block1(x1)
        x3 = self.block2(x2)
        x4 = self.block3(x3)
        x5 = self.block4(x4)
        x6 = self.flat(x5)
        x7 = self.act(self.fc1(x6))

        return x7

    def get_cnn_block(self, in_channels, out_channels, kernel_size, last_block=False):

        if last_block:
            # No pooling layer
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                nn.ReLU()
            )

        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

# Actual model used in this project
class cnn_80_encoder(nn.Module):

    def __init__(self):
        super().__init__()

        image_size = (80, 80, 3)
        num_channels = image_size[-1]

        self.block1 = self.get_cnn_block(num_channels, 64, 10)
        self.block2 = self.get_cnn_block(64, 128, 7)
        self.block3 = self.get_cnn_block(128, 128, 4)
        self.block4 = self.get_cnn_block(128, 256, 4, last_block=True)
        self.flat = nn.Flatten(start_dim=1, end_dim=-1) # start_dim=1 Do not flatten batch fim
        self.fc1 = nn.Linear(1024, 512)
        self.act = nn.Sigmoid()

    def forward(self, x1):
        x2 = self.block1(x1)
        x3 = self.block2(x2)
        x4 = self.block3(x3)
        x5 = self.block4(x4)
        x6 = self.flat(x5)
        x7 = self.act(self.fc1(x6))

        return x7

    def get_cnn_block(self, in_channels, out_channels, kernel_size, last_block=False):

        if last_block:
            # No pooling layer
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                nn.ReLU()
            )

        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(out_channels),
        )


class siamese_network(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.model = model
        self.fc = nn.Linear(512, 1) #unused
        self.act = nn.Sigmoid() #unused

        # Initialize all layer weights
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            print(module)
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.5, std=0.01)

        if isinstance(module, nn.Linear):
            print(module)
            module.weight.data.normal_(mean=0.0, std=0.2)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.5, std=0.01)

    def dist_abs(self, x1, x2):
        return torch.abs(x1 - x2)

    def forward(self, x1, x2):
        enc_1 = self.model(x1)
        enc_2 = self.model(x2)

        return enc_1, enc_2



# Unit testing
if __name__ == '__main__':

    print('Unit testing siamese network .. .. ..')

    net1 = cnn_80_encoder()

    image_in = torch.tensor(torch.zeros(size=(1,3,80,80)))

    out1 = net1(image_in)

    print(out1.shape)
