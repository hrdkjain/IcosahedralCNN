import torch.nn as nn
from icocnn.ico_conv import IcoConvS2R
from icocnn.ico_conv import IcoConvR2R
from icocnn.ico_conv import IcoBatchNorm


class IcoConvNet_Test(nn.Module):

    def __init__(self):
        super(IcoConvNet_Test, self).__init__()

        self.convolutional = IcoConvS2R(
                in_features=1,
                out_features=8,
                stride=2,
                subdivisions=5
        )

    def forward(self, x):
        x = self.convolutional(x)
        return x


class IcoConvNet_Test1(nn.Module):

    def __init__(self, corner_mode):
        super(IcoConvNet_Test1, self).__init__()

        self.convolutional = nn.Sequential(
            IcoConvS2R(
                in_features=1,
                out_features=8,
                stride=2,
                subdivisions=4,
                corner_mode=corner_mode),
            IcoBatchNorm(8),
            nn.ReLU(inplace=False))

        self.linear = nn.Sequential(nn.Linear(in_features=30720, out_features=10))

    def forward(self, x):
        x = self.convolutional(x)
        x = self.linear(x.view(x.size(0), -1))
        return x


class IcoConvNet_OriginalR2R(nn.Module):

    def __init__(self, corner_mode):
        super(IcoConvNet_OriginalR2R, self).__init__()

        self.convolutional = nn.Sequential(
            IcoConvS2R(
                in_features=1,
                out_features=8,
                stride=1,
                subdivisions=4,
                corner_mode=corner_mode
            ),
            IcoBatchNorm(8),
            nn.ReLU(inplace=False),

            IcoConvR2R(
                in_features=8,
                out_features=16,
                stride=2,
                subdivisions=4,
                corner_mode=corner_mode
            ),
            IcoBatchNorm(16),
            nn.ReLU(inplace=False),

            IcoConvR2R(
                in_features=16,
                out_features=16,
                stride=1,
                subdivisions=3,
                corner_mode=corner_mode
            ),
            IcoBatchNorm(16),
            nn.ReLU(inplace=False),

            IcoConvR2R(
                in_features=16,
                out_features=24,
                stride=2,
                subdivisions=3,
                corner_mode=corner_mode
            ),
            IcoBatchNorm(24),
            nn.ReLU(inplace=False),

            IcoConvR2R(
                in_features=24,
                out_features=24,
                stride=1,
                subdivisions=2,
                corner_mode=corner_mode
            ),
            IcoBatchNorm(24),
            nn.ReLU(inplace=False),

            IcoConvR2R(
                in_features=24,
                out_features=32,
                stride=2,
                subdivisions=2,
                corner_mode=corner_mode
            ),
            IcoBatchNorm(32),
            nn.ReLU(inplace=False),

            IcoConvR2R(
                in_features=32,
                out_features=64,
                stride=1,
                subdivisions=1,
                corner_mode=corner_mode
            ),
            IcoBatchNorm(64),
            nn.ReLU(inplace=False)
        )

        embedding_height = 5 * (2 ** 1)
        embedding_width = 2 ** (1 + 1)

        self.pooling = nn.MaxPool3d(
            kernel_size=(6, embedding_height, embedding_width),
            stride=(6, 1, 1)
        )

        self.linear = nn.Sequential(
            # linear 1
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(inplace=False),
            # linear 2
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=False),
            # linear 3
            nn.BatchNorm1d(32),
            nn.Linear(in_features=32, out_features=10)
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = self.pooling(x[:, None, :, :, :])
        x = self.linear(x.view(x.size(0), -1))
        return x


class IcoConvNet_OriginalS2R(nn.Module):

    def __init__(self, corner_mode):
        super(IcoConvNet_OriginalS2R, self).__init__()

        self.convolutional = nn.Sequential(
            IcoConvS2R(
                in_features=1,
                out_features=20,
                stride=1,
                subdivisions=4,
                corner_mode=corner_mode
            ),
            IcoBatchNorm(20),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(
                kernel_size=(6, 1, 1),
                stride=(6, 1, 1)
            ),

            IcoConvS2R(
                in_features=20,
                out_features=40,
                stride=2,
                subdivisions=4,
                corner_mode=corner_mode
            ),
            IcoBatchNorm(40),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(
                kernel_size=(6, 1, 1),
                stride=(6, 1, 1)
            ),

            IcoConvS2R(
                in_features=40,
                out_features=40,
                stride=1,
                subdivisions=3,
                corner_mode=corner_mode
            ),
            IcoBatchNorm(40),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(
                kernel_size=(6, 1, 1),
                stride=(6, 1, 1)
            ),

            IcoConvS2R(
                in_features=40,
                out_features=60,
                stride=2,
                subdivisions=3,
                corner_mode=corner_mode
            ),
            IcoBatchNorm(60),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(
                kernel_size=(6, 1, 1),
                stride=(6, 1, 1)
            ),

            IcoConvS2R(
                in_features=60,
                out_features=60,
                stride=1,
                subdivisions=2,
                corner_mode=corner_mode
            ),
            IcoBatchNorm(60),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(
                kernel_size=(6, 1, 1),
                stride=(6, 1, 1)
            ),

            IcoConvS2R(
                in_features=60,
                out_features=80,
                stride=2,
                subdivisions=2,
                corner_mode=corner_mode
            ),
            IcoBatchNorm(80),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(
                kernel_size=(6, 1, 1),
                stride=(6, 1, 1)
            ),

            IcoConvS2R(
                in_features=80,
                out_features=160,
                stride=1,
                subdivisions=1,
                corner_mode=corner_mode
            ),
            IcoBatchNorm(160),
            nn.ReLU(inplace=False)
        )

        embedding_height = 5 * (2 ** 1)
        embedding_width = 2 ** (1 + 1)

        self.pooling = nn.MaxPool3d(
            kernel_size=(6, embedding_height, embedding_width),
            stride=(6, 1, 1)
        )

        self.linear = nn.Sequential(
            # linear 1
            nn.BatchNorm1d(160),
            nn.Linear(in_features=160, out_features=64),
            nn.ReLU(inplace=False),
            # linear 2
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=False),
            # linear 3
            nn.BatchNorm1d(32),
            nn.Linear(in_features=32, out_features=10)
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = self.pooling(x[:, None, :, :, :])
        x = self.linear(x.view(x.size(0), -1))
        return x
