import torch
import torch.nn as nn
import torch.nn.functional as F

class LargeFOV(nn.Module):
    def __init__(self, n_class=21):
        super(LargeFOV, self).__init__()
        self.conv1_1=nn.Conv2d(n_class, 96, 3, stride=1, padding=1)
        self.conv1_2=nn.Conv2d(96, 128, 3, stride=1, padding=1)
        self.conv1_3=nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv2_1=nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv2_2=nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3_1=nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv3_2=nn.Conv2d(512, 1, 3, stride=1, padding=1)

    def forward(self, x):
        res = F.relu(self.conv1_1(x))
        res = F.relu(self.conv1_2(res))
        res = F.relu(self.conv1_3(res))
        res = F.max_pool2d(res, 2, stride=2)
        res = F.relu(self.conv2_1(res))
        res = F.relu(self.conv2_2(res))
        res = F.max_pool2d(res, 2, stride=2)
        res = F.relu(self.conv3_1(res))
        res = self.conv3_2(res)
        res = F.avg_pool2d(res, kernel_size=(res.shape[2], res.shape[3]))
        res = res.view((res.shape[0],res.shape[1]))
        # out = F.softmax(res)
        return res


class StanfordBNet(nn.Module):
    def __init__(self, n_class=8):
        super(StanfordBNet, self).__init__()

        self.conv1_1 = nn.Conv2d(n_class, 64, 5, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(3, 16, 5, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(16, 64, 5, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv3_4 = nn.Conv2d(512, 2, 3, stride=1, padding=1)

    def forward(self, segmented_input, original_input):
        res1 = F.relu(self.conv1_1(segmented_input))

        res2 = F.relu(self.conv2_1(original_input))
        res2 = F.max_pool2d(res2, 2, stride=1, padding=1)
        res2 = F.relu(self.conv2_2(res2))
        res2 = F.max_pool2d(res2, 2, stride=1, padding=1)

        res3 = torch.cat((res1, res2), 1)

        res3 = F.relu(self.conv3_1(res3))
        res3 = F.max_pool2d(res3, 2, stride=1)
        res3 = F.relu(self.conv3_2(res3))
        res3 = F.max_pool2d(res3, 2, stride=1)
        res3 = F.relu(self.conv3_3(res3))
        res3 = self.conv3_4(res3)
        # return res
        out = F.avg_pool2d(res3, (res3.shape[2],res3.shape[3]))
        out = F.softmax(out)
        n , _ , _ ,_  = segmented_input.size()
        return out.view(n,-1).transpose(0,1)[0]