import torch


class Model(torch.nn.Module):
    def __init__(self, channels):
        super(Model, self).__init__()
        self.channels = channels

        self.inter_channels1 = 16
        self.inter_channels2 = 64
        self.inter_channels3 = 128
        self.inter_channels4 = 256

        self.input_conv2d = torch.nn.Conv2d(in_channels=self.channels, out_channels=self.inter_channels1,
                                            kernel_size=(3, 3), stride=(1, 1),
                                            padding=(1, 1), bias=False)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.conv1 = torch.nn.Conv2d(in_channels=self.inter_channels1, out_channels=self.inter_channels2,
                                     kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1), bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=self.inter_channels2, out_channels=self.inter_channels3,
                                     kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1), bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=self.inter_channels3, out_channels=self.inter_channels3,
                                     kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1), bias=False)
        self.conv4 = torch.nn.Conv2d(in_channels=self.inter_channels3, out_channels=self.inter_channels4,
                                     kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1), bias=False)
        self.conv5 = torch.nn.Conv2d(in_channels=self.inter_channels4, out_channels=self.inter_channels4,
                                     kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1), bias=False)

        self.av_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.linear = torch.nn.Linear(self.inter_channels4, 4)

    def forward(self, x):
        out = self.input_conv2d(x)
        out = self.max_pool(out)
        out = torch.nn.functional.relu(out)

        out = self.conv1(out)
        out = torch.nn.functional.relu(out)

        out = self.conv2(out)
        out = torch.nn.functional.relu(out)

        out = self.conv3(out)
        out = torch.nn.functional.relu(out)

        out = self.conv4(out)
        out = torch.nn.functional.relu(out)

        out = self.conv5(out)
        out = torch.nn.functional.relu(out)

        out = self.av_pool(out)
        out = torch.squeeze(out)
        out = self.linear(out)
        return out

