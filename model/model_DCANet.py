import torch
import torch.nn as nn
from model.basic import DyfConv


class SimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-5):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class Res_SimAM_block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_SimAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.conv2 = nn.Sequential(self.conv2,SimAM(out_channels))
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

        self.simam = SimAM(out_channels)



    def forward(self, x):

        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.simam(out)

        out += residual
        out = self.relu(out)

        return out


class Res_SimAM_block_D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_SimAM_block_D, self).__init__()
        self.conv1 = DyfConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.conv2 = nn.Sequential(self.conv2,SimAM(out_channels))
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

        self.simam = SimAM(out_channels)



    def forward(self, x):

        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.simam(out)

        out += residual
        out = self.relu(out)

        return out


class DCANet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_SimAM_block, num_blocks=[3, 4, 6, 3], nb_filter=[16,32, 64, 128, 256], deep_supervision=True, mode='test', conv='usual'):
        super(DCANet, self).__init__()
        self.mode = mode
        self.relu = nn.ReLU(inplace=True)
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        if conv == 'Dual':
            self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
            self.conv2_0 = self._make_layer(Res_SimAM_block_D, nb_filter[1], nb_filter[2], num_blocks[1])
        elif conv == 'usual':
            self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
            self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3])

        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2], nb_filter[3], num_blocks[2])

        self.conv4_1 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3])


        self.conv2_2 = self._make_layer(block, nb_filter[2] * 2 + nb_filter[3], nb_filter[2])
        self.conv3_2 = self._make_layer(block, nb_filter[3] * 2 + nb_filter[4] + nb_filter[2], nb_filter[3],
                                        num_blocks[0])

        self.conv1_2 = self._make_layer(block, nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_3 = self._make_layer(block, nb_filter[2] * 3 + nb_filter[3] + nb_filter[1], nb_filter[2],
                                        num_blocks[0])

        self.conv0_2 = self._make_layer(block, nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1] * 3 + nb_filter[2] + nb_filter[0], nb_filter[1],
                                        num_blocks[0])

        self.conv0_3 = self._make_layer(block, nb_filter[0] * 3 + nb_filter[1], nb_filter[0])  #

        self.conv0_4_final = self._make_layer(block, nb_filter[0] * 5, nb_filter[0])

        #        self.conv0_4 = self._make_layer(block, nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        #        self.conv0_4_final = self._make_layer(block, nb_filter[0]*5, nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.down(x1_1)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_2)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_2)], 1))

        x4_1 = self.conv4_1(self.down(x3_1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.up(x4_1), self.down(x2_2)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up(x3_2), self.down(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_3), self.down(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_3)], 1))

        Final_x0_4 = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(x4_1)), self.up_8(self.conv0_3_1x1(x3_2)),
                       self.up_4(self.conv0_2_1x1(x2_3)), self.up(self.conv0_1_1x1(x1_3)), x0_3], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1).sigmoid()
            output2 = self.final2(x0_2).sigmoid()
            output3 = self.final3(x0_3).sigmoid()
            output4 = self.final4(Final_x0_4).sigmoid()
            if self.mode == 'train':
                return [output1, output2, output3, output4]
            else:
                return output4
        else:
            output = self.final(Final_x0_4).sigmoid()
            return output


class DCANet_L(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block=Res_SimAM_block, num_blocks=[3, 4, 6, 3], nb_filter=[16,32, 64, 128, 256], deep_supervision=True):
        super(DCANet_L, self).__init__()
        self.mode = mode
        self.relu = nn.ReLU(inplace=True)
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        # 第一列
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3])
        # 第二列

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv4_1 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[0])

        self.conv3_2 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv2_1 = self._make_layer(block,  nb_filter[3]+nb_filter[2], nb_filter[2])

        self.conv1_1 = self._make_layer(block, nb_filter[2] + nb_filter[1], nb_filter[1])

        self.conv0_1 = self._make_layer(block,  nb_filter[1] + nb_filter[0], nb_filter[0])

        self.conv0_4_final = self._make_layer(block, nb_filter[0] * 5, nb_filter[0])


        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x4_1 = self.conv4_1(self.down(x3_1))

        x3_2 = self.conv3_2(torch.cat([x3_1, self.up(x4_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_2)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_1)], 1))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_1)], 1))
        output = self.final(x0_1)
        return output

class DCANet_M(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_SimAM_block, num_blocks=[3, 4, 6, 3], nb_filter=[16, 32, 64, 128, 256], deep_supervision=True):
        super(DCANet_M, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3])

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1], nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2], nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])

        self.conv4_1 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3])


        self.conv2_2 = self._make_layer(block, nb_filter[2], nb_filter[2])
        self.conv3_2 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3],
                                        num_blocks[0])

        self.conv1_2 = self._make_layer(block, nb_filter[1], nb_filter[1])
        self.conv2_3 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2],
                                        num_blocks[0])

        self.conv0_2 = self._make_layer(block, nb_filter[0], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1],
                                        num_blocks[0])

        self.conv0_3 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])  #

        self.conv0_4_final = self._make_layer(block, nb_filter[0] * 5, nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(x0_0)

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(x1_0)

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(x2_0)

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(x2_1)
        x1_2 = self.conv1_2(x1_1)
        x0_2 = self.conv0_2(x0_1)

        x4_1 = self.conv4_1(self.down(x3_1))
        x3_2 = self.conv3_2(torch.cat([x3_1, self.up(x4_1)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_2, self.up(x3_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_2, self.up(x2_3)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_2, self.up(x1_3)], 1))

        Final_x0_4 = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(x4_1)), self.up_8(self.conv0_3_1x1(x3_2)),
                       self.up_4(self.conv0_2_1x1(x2_3)), self.up(self.conv0_1_1x1(x1_3)), x0_3], 1))


        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(Final_x0_4)
            if self.mode == 'train':
                return [output1, output2, output3, output4]
            else:
                return output4
        else:
            output = self.final(Final_x0_4)
            return output


class DCANet_R(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter, deep_supervision=False):
        super(DCANet_R, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3])

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2], nb_filter[3], num_blocks[2])

        self.conv4_1 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3])


        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_2 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3],
                                        num_blocks[0])

        self.conv1_2 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_3 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2],
                                        num_blocks[0])

        self.conv0_2 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1],
                                        num_blocks[0])

        self.conv0_3 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])  #

        self.conv0_4_final = self._make_layer(block, nb_filter[0] * 5, nb_filter[0])


        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(x0_0)

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.down(x0_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.down(x1_1)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_1, self.up(x3_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_1, self.up(x2_2)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_1, self.up(x1_2)], 1))

        x4_1 = self.conv4_1(self.down(x3_1))
        x3_2 = self.conv3_2(torch.cat([x3_1, self.up(x4_1)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_2, self.up(x3_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_2, self.up(x2_3)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_2, self.up(x1_3)], 1))

        Final_x0_4 = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(x4_1)), self.up_8(self.conv0_3_1x1(x3_2)),
                       self.up_4(self.conv0_2_1x1(x2_3)), self.up(self.conv0_1_1x1(x1_3)), x0_3], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(Final_x0_4)
            if self.mode == 'train':
                return [output1, output2, output3, output4]
            else:
                return output4
        else:
            output = self.final(Final_x0_4)
            return output
