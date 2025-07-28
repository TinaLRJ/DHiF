import timm 
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

class MSB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSB, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=7, dilation=7, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(4 * out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class CBAM(nn.Module):

    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class GAM_Attention(nn.Module):

    def __init__(self, in_channels, out_channels, rate=4):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        inchannel_rate = int(in_channels / rate)

        self.linear1 = nn.Linear(in_channels, inchannel_rate)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(inchannel_rate, in_channels)

        self.conv1 = nn.Conv2d(in_channels,
                               inchannel_rate,
                               kernel_size=7,
                               padding=3,
                               padding_mode='replicate')

        self.conv2 = nn.Conv2d(inchannel_rate,
                               out_channels,
                               kernel_size=7,
                               padding=3,
                               padding_mode='replicate')

        self.norm1 = nn.BatchNorm2d(inchannel_rate)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.linear2(self.relu(self.linear1(x_permute))).view(
            b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.relu(self.norm1(self.conv1(x)))
        x_spatial_att = self.sigmoid(self.norm2(self.conv2(x_spatial_att)))

        out = x * x_spatial_att

        return out

class Res_CBAM_block(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=stride), nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        self.cbam = CBAM(out_channels)
        self.gam = GAM_Attention(in_channels, out_channels)

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)  
        out = self.relu(out)
        return out

class ASPP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ASPP, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, dilation=5, padding=5, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, dilation=7, padding=7, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        x_mean = x.mean(dim=(2, 3), keepdim=True)
        x_upsampled = F.interpolate(x_mean, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv5 = self.conv5(x_upsampled)
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), dim=1))

class Compression_Layer(nn.Module):
    def __init__(self, out_c):
        super(Compression_Layer,self).__init__()
        self.c5_down = ASPP(in_dim=2048,out_dim=out_c)
        self.c4_down = nn.Sequential(
            nn.Conv2d(1024, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.c3_down = nn.Sequential(
            nn.Conv2d(512, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.c2_down = nn.Sequential(
            nn.Conv2d(256, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.c1_down = nn.Sequential(
            nn.Conv2d(64, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 5
        c1, c2, c3, c4, c5 = xs
        c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        return c5, c4, c3, c2, c1
    

class SAFB(nn.Module):
    def __init__(self, in_dim, m_size):
        super(SAFB, self).__init__()
        self.conv_l_pre_down = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )
        self.conv_l_post_down = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )

        self.trans = MSB(128,2)

    def forward(self, l, m, return_feats=False):
        tgt_size = m.shape[2:]
        l = self.conv_l_pre_down(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        l = self.conv_l_post_down(l)
        m = self.conv_m(m)
        attn = self.trans(torch.cat([l, m], dim=1))
        attn_l, attn_m= torch.softmax(attn, dim=1).chunk(2, dim=1)
        lm = attn_l * l + attn_m * m
        if return_feats:
            return lm, dict(attn_l=attn_l, attn_m=attn_m, l=l, m=m)
        return lm
    
class MAFNet(nn.Module):
    def __init__(self, base_size=256):
        super(MAFNet,self).__init__()
        self.shared_encoder = timm.create_model(model_name="resnet18", pretrained=False, in_chans=1, features_only=True)
        # checkpoint_path = r"pretrain model path"
        # state_dict = torch.load(checkpoint_path)
        # self.shared_encoder.load_state_dict(state_dict, False)

        self.translayer = Compression_Layer(out_c=64)
        self.merge_layers = nn.ModuleList([SAFB(in_dim=64,m_size=m_size) for m_size in (base_size//32, base_size//16, base_size//8, base_size//4, base_size//2)])

        self.up = nn.Upsample(scale_factor=2,
                              mode='bilinear',
                              align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4,
                                mode='bilinear',
                                align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8,
                                mode='bilinear',
                                align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,
                                 mode='bilinear',
                                 align_corners=True)
        self.up_32 = nn.Upsample(scale_factor=32,
                            mode='bilinear',
                            align_corners=True)
        self.res_cbam = Res_CBAM_block(in_channels=64*5, out_channels=64)
        self.rf_1 = MSB(64,64)
        self.rf_2 = MSB(32,32)
        self.out_layer_00 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.out_layer_01 = nn.Conv2d(32, 1, 1)

    
    def ms_resize(self,img, scales=[1,2], base_h=None, base_w=None, interpolation=InterpolationMode.BILINEAR):
        assert isinstance(scales, (list, tuple))
        if base_h is None and base_w is None:
            h, w = img.shape[-2], img.shape[-1]
        else:
            h, w = base_h, base_w
        resized_images = []
        for s in scales:
            new_h = int(s * h // 32 * 32)
            new_w = int(s * w // 32 * 32)
            resized_img = TF.resize(img, [new_h, new_w], interpolation=interpolation)
            resized_images.append(resized_img)
        return resized_images
    
    def encoder_translayer(self, x):  
        en_feats = self.shared_encoder(x)
        trans_feats = self.translayer(en_feats)
        return trans_feats
    def forward(self, x):
        m_scale, l_scale = self.ms_resize(x)  
        m_trans_feats = self.encoder_translayer(m_scale)
        l_trans_feats = self.encoder_translayer(l_scale)
        feats = []
        for l, m, layer in zip(l_trans_feats, m_trans_feats,self.merge_layers):
            siu_outs = layer(l=l, m=m)
            feats.append(siu_outs)
        feat = torch.cat((self.up_32(feats[0]),
                          self.up_16(feats[1]),
                          self.up_8(feats[2]),
                          self.up_4(feats[3]),
                          self.up(feats[4])
                          ),dim=1)
        fuse_feat = self.res_cbam(feat)
        pred = self.out_layer_01(self.rf_2(self.out_layer_00(self.rf_1(fuse_feat))))
        return pred.sigmoid()
    
net = MAFNet(base_size=256).cuda()
inputs = torch.randn(1, 3, 256, 256).cuda()
print(inputs.shape)
