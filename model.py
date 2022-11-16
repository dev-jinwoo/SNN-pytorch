import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetL1(nn.Module):
    def __init__(self):
        super(SiameseNetL1, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.conv5 = nn.Conv2d(256, 256, 4)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 1)

        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5_bn = nn.BatchNorm2d(256)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def sub_forward(self, x):

        out = F.max_pool2d(self.conv1_bn(F.relu(self.conv1(x))), 2)
        out = F.max_pool2d(self.conv2_bn(F.relu(self.conv2(out))), 2)
        out = F.max_pool2d(self.conv3_bn(F.relu(self.conv3(out))), 2)
        out = F.max_pool2d(self.conv4_bn(F.relu(self.conv4(out))), 2)
        out = self.conv5_bn(F.relu(self.conv5(out)))

        out = out.view(out.shape[0], -1)
        out = F.sigmoid(self.fc1(out))

        return out

    def forward(self, x1, x2):
        # encode image pairs
        h1 = self.sub_forward(x1)
        h2 = self.sub_forward(x2)

        # L1 loss
        diff = torch.abs(h1 - h2)
        scores = self.fc2(diff)

        return scores

    def test_forward(self, h1, h2):
        diff = torch.abs(h1 - h2)

        scores = self.fc2(diff)

        return scores


class SiameseNetL2(nn.Module):
    def __init__(self):
        super(SiameseNetL2, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.conv5 = nn.Conv2d(256, 256, 4)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 1)

        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5_bn = nn.BatchNorm2d(256)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def sub_forward(self, x):

        out = F.max_pool2d(self.conv1_bn(F.relu(self.conv1(x))), 2)
        out = F.max_pool2d(self.conv2_bn(F.relu(self.conv2(out))), 2)
        out = F.max_pool2d(self.conv3_bn(F.relu(self.conv3(out))), 2)
        out = F.max_pool2d(self.conv4_bn(F.relu(self.conv4(out))), 2)
        out = self.conv5_bn(F.relu(self.conv5(out)))

        out = out.view(out.shape[0], -1)
        out = F.sigmoid(self.fc1(out))

        return out

    def forward(self, x1, x2):
        # encode image pairs
        h1 = self.sub_forward(x1)
        h2 = self.sub_forward(x2)

        # L1 loss
        diff = torch.pow((h1 - h2), 2)
        scores = self.fc2(diff)

        return scores

    def test_forward(self, h1, h2):
        diff = torch.pow((h1 - h2), 2)

        scores = self.fc2(diff)

        return scores


class SiameseNetCos(nn.Module):
    def __init__(self):
        super(SiameseNetCos, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.conv5 = nn.Conv2d(256, 256, 4)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 1)

        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5_bn = nn.BatchNorm2d(256)

        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def sub_forward(self, x):

        out = F.max_pool2d(self.conv1_bn(F.relu(self.conv1(x))), 2)
        out = F.max_pool2d(self.conv2_bn(F.relu(self.conv2(out))), 2)
        out = F.max_pool2d(self.conv3_bn(F.relu(self.conv3(out))), 2)
        out = F.max_pool2d(self.conv4_bn(F.relu(self.conv4(out))), 2)
        out = self.conv5_bn(F.relu(self.conv5(out)))

        out = out.view(out.shape[0], -1)
        out = self.fc1(out)

        return out

    def forward(self, x1, x2):
        # encode image pairs
        h1 = self.sub_forward(x1)
        h2 = self.sub_forward(x2)

        # cosine similarity
        diff = self.cos(h1, h2)
        scores = torch.unsqueeze(diff, 1)

        return scores


if __name__ == '__main__':
    model = SiameseNetL1()

    x = torch.randn((1, 1, 210, 210))
    y = torch.randn((1, 1, 210, 210))

    result = model(x, x)

    print(result)






