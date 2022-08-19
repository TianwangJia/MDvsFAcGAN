import torch.nn as nn
import torch.nn.functional as F
import torch

# 定义net_block调用, 包括Conv, Bn, LeakyReLU
class net_block(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, dilation, padding, is_bn=True, is_relu=True):
        super(net_block, self).__init__()
        # 空洞卷积计算公式: [x+2p-k-(k-1)*(d-1)]/s + 1,中括号表示向下取整
        layers = [nn.Conv2d(in_filters, out_filters, kernel_size, dilation=dilation, padding=padding)]
        if is_bn:
            layers.append(nn.BatchNorm2d(out_filters))
        if is_relu:
            layers.append(nn.LeakyReLU(0.2))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Generator1_MD(nn.Module):
    def __init__(self, channel_num=64):
        super(Generator1_MD, self).__init__()

        #TODO: 比论文前面多了一层, chn-chn
        self.net1 = net_block(1,channel_num, kernel_size=3, dilation=1, padding=1)
        self.net2 = net_block(channel_num,channel_num, kernel_size=3, dilation=1, padding=1)
        self.net3 = net_block(channel_num,channel_num*2, kernel_size=3, dilation=2, padding=2)
        self.net4 = net_block(channel_num*2,channel_num*4, kernel_size=3, dilation=4, padding=4)
        self.net5 = net_block(channel_num*4,channel_num*8, kernel_size=3, dilation=8, padding=8)
        self.net6 = net_block(channel_num*8,channel_num*4, kernel_size=3, dilation=4, padding=4)
        self.net7 = net_block(channel_num*4,channel_num*2, kernel_size=3, dilation=2, padding=2)
        self.net8 = net_block(channel_num*2,channel_num, kernel_size=3, dilation=1, padding=1)
        
        self.out = net_block(channel_num,1, kernel_size=1,dilation=1, padding=0, is_bn=False, is_relu=False)

    def forward(self, input):
        '''
        in: [B,1,128,128]
        out: [B,1,128,128]
        '''
        x = self.net1(input) #[B,chn,128,128]
        x = self.net2(x) #[B,chn,128,128]
        x = self.net3(x) #[B,chn*2,128,128]
        x = self.net4(x) #[B,chn*4,128,128]
        x = self.net5(x) #[B,chn*8,128,128]
        x = self.net6(x) #[B,chn*4,128,128]
        x = self.net7(x) #[B,chn*2,128,128]
        x = self.net8(x) #[B,chn,128,128]

        out = self.out(x) #[B,1,128,128]

        return out

class Generator2_FA(nn.Module):
    def __init__(self, channel_num=64):
        super(Generator2_FA, self).__init__()

        self.net1 = net_block(1,channel_num, kernel_size=3, dilation=1, padding=1)
        self.net2 = net_block(channel_num,channel_num, kernel_size=3, dilation=2, padding=2)
        self.net3 = net_block(channel_num,channel_num, kernel_size=3, dilation=4, padding=4)
        self.net4 = net_block(channel_num,channel_num, kernel_size=3, dilation=8, padding=8)
        self.net5 = net_block(channel_num,channel_num, kernel_size=3, dilation=16, padding=16)
        self.net6 = net_block(channel_num,channel_num, kernel_size=3, dilation=32, padding=32)
        self.net7 = net_block(channel_num,channel_num, kernel_size=3, dilation=64, padding=64)
        self.net8 = net_block(channel_num,channel_num, kernel_size=3, dilation=32, padding=32)
        self.net9 = net_block(channel_num*2,channel_num, kernel_size=3, dilation=16, padding=16)
        self.net10 = net_block(channel_num*2,channel_num, kernel_size=3, dilation=8, padding=8)
        self.net11 = net_block(channel_num*2,channel_num, kernel_size=3, dilation=4, padding=4)
        self.net12 = net_block(channel_num*2,channel_num, kernel_size=3, dilation=2, padding=2)
        self.net13 = net_block(channel_num*2,channel_num, kernel_size=3, dilation=1, padding=1)
        
        self.out = net_block(channel_num,1, kernel_size=3, dilation=1, padding=1, is_bn=False, is_relu=False)

    def forward(self, input):
        '''
        in: [B,1,128,128]
        out: [B,1,128,128]
        '''
        #TODO: 比论文少了一个跳连, x1到x13的
        x1 = self.net1(input) # [B,chn,128,128]
        x2 = self.net2(x1) # [B,chn,128,128]
        x3 = self.net3(x2) # [B,chn,128,128]
        x4 = self.net4(x3) # [B,chn,128,128]
        x5 = self.net5(x4) # [B,chn,128,128]
        x6 = self.net6(x5) # [B,chn,128,128]
        x7 = self.net7(x6) # [B,chn,128,128]
        x8 = self.net8(x7) # [B,chn,128,128] 

        x9 = self.net9(torch.cat((x6,x8), dim=1)) # [B,chn,128,128]
        x10 = self.net10(torch.cat((x5,x9),dim=1)) # [B,chn,128,128]
        x11 = self.net11(torch.cat((x4,x10),dim=1)) # [B,chn,128,128]
        x12 = self.net12(torch.cat((x3,x11),dim=1)) # [B,chn,128,128]
        x13 = self.net13(torch.cat((x2,x12), dim=1)) # [B,chn,128,128]

        out = self.out(x13) # [B,1,128,128]

        return out


class Discriminator(nn.Module):
    def __init__(self, batch_size=16) -> None:
        super(Discriminator, self).__init__()

        self.batch_size = batch_size

        self.maxpool1 = nn.MaxPool2d(kernel_size=[2,2])
        self.maxpool2 = nn.MaxPool2d(kernel_size=[2,2])

        self.net1 = net_block(2,24,kernel_size=3, dilation=1, padding=1)
        self.net2 = net_block(24,24,kernel_size=3, dilation=1, padding=1)
        self.net3 = net_block(24,24,kernel_size=3, dilation=1, padding=1)
        self.net4 = net_block(24,1,kernel_size=3, dilation=1, padding=1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(3)

        self.Tanh1 = nn.Tanh()
        self.Tanh2 = nn.Tanh()

        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

        self.Softmax = nn.Softmax()

    def forward(self, input):
        '''
        in: [3B,2,128,128]
        out: [B,3]*3, 1
        '''
        size = int(input.shape[0]/3)
        x = self.maxpool1(input) # [3B,2,64,64]
        x = self.maxpool2(x) # [3B,2,32,32]

        x = self.net1(x) # [3B,24,32,32]
        x = self.net2(x) # [3B,24,32,32]
        x = self.net3(x) # [3B,24,32,32]
        x = self.net4(x) # [3B,24,32,32]

        conv_fmap = x # 用于计算Generator consistency loss的D fmap

        x = x.view(-1,1024) # flatten [3B,1024]
        x = self.fc1(x) # [3B,128]
        x = x.unsqueeze(2).unsqueeze(3) 
        x = self.bn1(x) 
        x = self.Tanh1(x) # [3B,128,1,1]

        x = x.view(-1,128) # flatten [3B,128]
        x = self.fc2(x) # [3B,64]
        x = x.unsqueeze(2).unsqueeze(3) 
        x = self.bn2(x) 
        x = self.Tanh2(x) # [3B,64,1,1]

        x = x.view(-1,64) # flatten [3B,1024]
        x = self.fc3(x) # [3B,3]
        x = x.unsqueeze(2).unsqueeze(3) 
        x = self.bn3(x) 
        x = self.Softmax(x) # [3B,3,1,1]
        x = x.squeeze(3).squeeze(2) # [3B,3]

        realscore0, realscore1, realscore2 = torch.split(x, size, dim=0) # 真实得分
        real, feat1, feat2 = torch.split(conv_fmap, size, dim=0)
        # fakeDist = torch.mean(torch.pow(feat1 - feat2, 2).view(-1,32*32),dim=1,keepdim=True) # 用于计算Generator consistency loss的feat G1 G2的L2距离
        # 计算每张图距离 返回[B,1]
        fakeDist = torch.mean(torch.pow(feat1 - feat2, 2)) # 用于计算Generator consistency loss的feat G1 G2的L2距离
        # 返回标量 代表论文中的 Generator consistency loss
        return realscore0, realscore1, realscore2, fakeDist



if __name__ == '__main__':
    batch_size = 2
    G1 = Generator1_MD()
    G2 = Generator2_FA()
    D = Discriminator(batch_size=batch_size)

    in_1 = torch.randn(batch_size,1,128,128)
    out_1 = G1(in_1)
    out_2 = G1(in_1)

    temp  = torch.randn(3*batch_size,2,128,128)

    realscore0, realscore1, realscore2, fakeDist = D(temp)

    print('out_1:',out_1.shape)
    print('out_2:',out_2.shape)
    print('r_0:', realscore0.shape)
    print('r_1:', realscore1.shape)
    print('r_2:', realscore2.shape)
    print('fakeDist:',fakeDist)

