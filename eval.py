from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from models import *
from datasets import *
from utils import *

import os

# 获得指定测试集的 F1, precision, recall
def get_score(G1, G2, dataloader, device):
    G1.eval()
    G2.eval()

    sum_g1_F1 = 0
    sum_g2_F1 = 0
    sum_g_F1 = 0
    sum_g1_prec = 0
    sum_g2_prec = 0
    sum_g_prec = 0
    sum_g1_recall = 0
    sum_g2_recall = 0
    sum_g_recall = 0

    for i, batch_data in enumerate(tqdm(dataloader)):
        org_img = batch_data['org'].to(device)
        gt_img = batch_data['gt'].to(device)

        g1_out = G1(org_img)
        g2_out = G2(org_img)
        # 将G的输出约束到 0-1之间
        g1_out = torch.clamp(g1_out, 0.0, 1.0)
        g2_out = torch.clamp(g2_out, 0.0, 1.0)
        g_out = (g1_out + g2_out) / 2 # 取均值的方式进行融合

        g1_out = g1_out.detach().cpu().numpy()
        g2_out = g2_out.detach().cpu().numpy()
        g_out = g_out.detach().cpu().numpy()
        gt_img = gt_img.cpu().numpy()

        val_g1_prec, val_g1_recall, val_g1_F1 = calculateF1Measure(g1_out, gt_img, 0.5)
        val_g2_prec, val_g2_recall, val_g2_F1 = calculateF1Measure(g2_out, gt_img, 0.5)
        val_g_prec, val_g_recall, val_g_F1 = calculateF1Measure(g_out, gt_img, 0.5)
        
        sum_g1_F1 = sum_g1_F1 + val_g1_F1
        sum_g2_F1 = sum_g2_F1 + val_g2_F1
        sum_g_F1 = sum_g_F1 + val_g_F1

        sum_g1_prec = sum_g1_prec + val_g1_prec
        sum_g2_prec = sum_g2_prec + val_g2_prec
        sum_g_prec = sum_g_prec + val_g_prec

        sum_g1_recall = sum_g1_recall + val_g1_recall
        sum_g2_recall = sum_g2_recall + val_g2_recall
        sum_g_recall = sum_g_recall + val_g_recall
    
    sum_g1_F1 = sum_g1_F1/len(dataloader)
    sum_g2_F1 = sum_g2_F1/len(dataloader)
    sum_g_F1 = sum_g_F1/len(dataloader)

    sum_g1_prec = sum_g1_prec/len(dataloader)
    sum_g2_prec = sum_g2_prec/len(dataloader)
    sum_g_prec = sum_g_prec/len(dataloader)

    sum_g1_recall = sum_g1_recall/len(dataloader)
    sum_g2_recall = sum_g2_recall/len(dataloader)
    sum_g_recall = sum_g_recall/len(dataloader)

    return (sum_g_F1, sum_g_prec, sum_g_recall), (sum_g1_F1, sum_g1_prec, sum_g1_recall), (sum_g2_F1, sum_g2_prec, sum_g2_recall)

# 保存模型输出图片
def get_image(G1, G2, dataloader, batch_size, device, save_path, specified_model='G'):
    '''
    specified_model: G, G1, G2 指定用哪个模型输出结果, G为 G1,G2 平均融合
    '''
    for i, batch_data in enumerate(tqdm(dataloader)):
        org_img = batch_data['org'].to(device)
        gt_img = batch_data['gt'].to(device)

        g1_out = G1(org_img)
        g2_out = G2(org_img)
        # 将G的输出约束到 0-1之间
        g1_out = torch.clamp(g1_out, 0.0, 1.0)
        g2_out = torch.clamp(g2_out, 0.0, 1.0)

        if specified_model == 'G':
            output_image = (g1_out+g2_out)/2
        elif specified_model == 'G1':
            output_image = g1_out
        elif specified_model == 'G2':
            output_image = g2_out
        
        org_img = make_grid(org_img, nrow=batch_size, normalize=True) 
        gt_img = make_grid(gt_img, nrow=batch_size, normalize=True)
        output_image = make_grid(output_image, nrow=batch_size, normalize=True)

        img_sample = torch.cat((org_img, gt_img, output_image), 1) # 上下拼接

        save_image(img_sample, "%s/%s.png"%(save_path,str(i+1)), normalize=True)





if __name__ == '__main__':
    train_path = './data/MDvsFA_train'
    test_path = './data/test'
    img_size = 128
    batch_size = 5
    model_path = './save_model'
    is_get_score = True # 是否获得F1分数
    is_get_image = True # 是否获得输出图片
    output_path = './'
    specified_model = 'G' # 指定输出模型

    # Datasets and DataLoader
    org_transform = [
        transforms.CenterCrop(size = img_size),
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5,),(0.5,)),
    ]
    gt_transform = [
        transforms.CenterCrop(size = img_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)),
    ]
    test_dataset_MDvsFA = InfraredImageDataset(train_path, test_path, org_transform, gt_transform, mode='test', test_data='MDvsFA')
    test_dataset_Sirst = InfraredImageDataset(train_path, test_path, org_transform, gt_transform, mode='test', test_data='Sirst')
    MDvsFA_dataloader = DataLoader(
        test_dataset_MDvsFA,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    Sirst_dataloader = DataLoader(
        test_dataset_Sirst,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G1 = Generator1_MD(channel_num=64).to(device)
    G2 = Generator2_FA(channel_num=64).to(device)
    G1.load_state_dict((torch.load("%s/G1.pth"%(model_path))))
    G2.load_state_dict((torch.load("%s/G2.pth"%(model_path))))

    if is_get_score:
        score_MDvsFA_G, score_MDvsFA_G1, score_MDvsFA_G2 = get_score(G1, G2, MDvsFA_dataloader, device)
        score_Sirst_G, score_Sirst_G1, score_Sirst_G2 = get_score(G1, G2, Sirst_dataloader, device)

        print('MDvsFA test F1: ', score_MDvsFA_G[0])
        print('Sirst test F1: ', score_Sirst_G[0])

    if is_get_image:
        os.makedirs("%s/eval_image/MDvsFA"%(output_path), exist_ok=True)
        os.makedirs("%s/eval_image/Sirst"%(output_path), exist_ok=True)

        MDvsFA_save_path = "%s/eval_image/MDvsFA"%(output_path)
        Sirst_save_path = "%s/eval_image/Sirst"%(output_path)

        get_image(G1, G2, MDvsFA_dataloader, batch_size, device, MDvsFA_save_path, specified_model)
        get_image(G1, G2, Sirst_dataloader, batch_size, device, Sirst_save_path, specified_model)
        

    

