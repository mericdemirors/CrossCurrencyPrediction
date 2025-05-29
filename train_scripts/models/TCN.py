import torch
import torch.nn as nn

class TCN(nn.Module):
    def __init__(self, input_features, output_features, output_window, dropout=0.2, device="cpu"):
        super(TCN, self).__init__()
        self.output_features = output_features
        self.output_window = output_window
        self.device = device

        # first part with the conv1d layers, this layers capture the intra-feature correlations
        self.conv1d_1 = nn.Conv1d(in_channels=input_features, out_channels=32, kernel_size=3, groups=input_features)
        self.bn1d_1 = nn.BatchNorm1d(32)
        self.drop_1 = nn.Dropout1d(dropout)
        self.conv1d_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, groups=32)
        self.bn1d_2 = nn.BatchNorm1d(64)
        self.drop_2 = nn.Dropout1d(dropout)
        self.conv1d_3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, groups=64)
        self.bn1d_3 = nn.BatchNorm1d(128)
        self.drop_3 = nn.Dropout1d(dropout)

        # second part with the conv2d layers, this layers capture the inter-feature correlations
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.bn2d_1 = nn.BatchNorm2d(32)
        self.conv2d_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.conv2d_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.bn2d_3 = nn.BatchNorm2d(128)
        self.conv2d_4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1,2))
        self.bn2d_4 = nn.BatchNorm2d(64)
        self.conv2d_5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), dilation=(1,2))
        self.bn2d_5 = nn.BatchNorm2d(32)
        self.conv2d_6 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(3, 3), dilation=(1,2))
        self.bn2d_6 = nn.BatchNorm2d(8)
        self.pool_6 = nn.AvgPool2d(kernel_size=(1,2))

        self.conv2d_7 = None
        # this is the initialization we want to make, but we are making this in the first forward loop to make it dynamic
        # self.conv2d_7 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(x.shape[2]-output_features+1, x.shape[3]-output_window+1))

        self.pad = nn.ReplicationPad2d((1,1,0,0))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1d_1(x)
        x = self.bn1d_1(x)
        x = self.relu(x)
        x = self.drop_1(x)

        x = self.pad(x)
        x = self.conv1d_2(x)
        x = self.bn1d_2(x)
        x = self.relu(x)
        x = self.drop_2(x)

        x = self.conv1d_3(x)
        x = self.bn1d_3(x)
        x = self.relu(x)
        x = self.drop_3(x)

        # [batch, coin_feature, time]
        x = x.unsqueeze(1).permute([0,1,3,2])
        # [batch, image_channel, time, coin_feature]

        x = self.conv2d_1(x)
        x = self.bn2d_1(x)
        x = self.relu(x)

        x = self.conv2d_2(x)
        x = self.bn2d_2(x)
        x = self.relu(x)

        x = self.conv2d_3(x)
        x = self.bn2d_3(x)
        x = self.relu(x)

        x = self.conv2d_4(x)
        x = self.bn2d_4(x)
        x = self.relu(x)

        x = self.conv2d_5(x)
        x = self.bn2d_5(x)
        x = self.relu(x)

        x = self.conv2d_6(x)
        x = self.bn2d_6(x)
        x = self.relu(x)
        x = self.pool_6(x)

        if self.conv2d_7:
            x = self.conv2d_7(x)
        else:
            # last layer is dynamically created depending on the input size, so we do it in the first forward pass
            self.conv2d_7 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(x.shape[2]-self.output_features+1, x.shape[3]-self.output_window+1)).to(self.device)
            x = self.conv2d_7(x)

        return torch.squeeze(x)
    
    def call(self, x, y):
        return self(x)