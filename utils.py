import imghdr
from locale import normalize
import torch
import numpy as np
from torchvision.utils import save_image, make_grid

# 初始化网络权重
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        # torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.weight.data)
        # 这里如果用 torch.nn.init.xavier_normal_会因为torchtext版本太高，不支持一维的词向量，仅仅支持二维以上的而报错
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)

# 计算F1分数
def calculateF1Measure(output_image,gt_image,thre):
    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    out_bin = output_image>thre
    gt_bin = gt_image>thre
    recall = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(gt_bin)) # 点乘
    prec   = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(out_bin))
    F1 = 2*recall*prec/np.maximum(0.001,recall+prec)
    return prec, recall, F1

# sample image to save
def sample_images(G1,G2,batches_done, test_data, batch_size, save_path, device):
    org_img = test_data['org'].to(device)
    gt_img = test_data['gt'].to(device)

    G1.eval()
    G2.eval()

    g1_out = G1(org_img)
    g2_out = G2(org_img)

    # 将G的输出约束到 0-1之间
    g1_out = torch.clamp(g1_out, 0.0, 1.0)
    g2_out = torch.clamp(g2_out, 0.0, 1.0)

    g_out = (g1_out+g2_out)/2

    # org_img = inverse_normalize(org_img, (0.5,), (0.5))
    # gt_img = inverse_normalize(gt_img, (0.5,), (0.5))
    # g_out = inverse_normalize(g_out, (0.5,), (0.5))

    # 拼成一副图像
    org_img = make_grid(org_img, nrow=batch_size, normalize=True) 
    gt_img = make_grid(gt_img, nrow=batch_size, normalize=True)
    g1_out = make_grid(g1_out, nrow=batch_size, normalize=True) 
    g2_out = make_grid(g2_out, nrow=batch_size, normalize=True) 
    g_out = make_grid(g_out, nrow=batch_size, normalize=True)

    img_sample = torch.cat((org_img, gt_img, g1_out, g2_out, g_out), 1) # 上下拼接

    save_image(img_sample, "%s/%s.png"%(save_path,batches_done), normalize=True)

