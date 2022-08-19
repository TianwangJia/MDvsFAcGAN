import argparse
from fileinput import hook_encoded
from logging.config import valid_ident
import os
from re import A
from tkinter import scrolledtext
import numpy as np
import math
import itertools
import sys
import datetime
import time
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
# from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn   
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs  of training")
parser.add_argument("--train_data_path", type=str, default="/root/autodl-tmp/data/MDvsFA_train", help="path to the train datasets")
parser.add_argument("--test_data_path", type=str,default="/root/autodl-tmp/data/test", help="path to the test datasets")
parser.add_argument("--output_path", type=str,default="/root/autodl-tmp/output", help="path to save model and outputs")
parser.add_argument("--log_path", type=str, default="/root/autodl-tmp/log", help="path to store the logs")
parser.add_argument("--pretrain_model_path", type=str, default='./pretrain_model', help="path to pretrained model")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--test_batch_size", type=int, default=5, help="size of the test batches")
parser.add_argument("--n_cpus", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=128, help="size of images")
parser.add_argument("--channel_num", type=int, default=64, help="number of basic channel of gan")
parser.add_argument("--lambda1", type=int, default=100, help="G1 data loss lambda1")
parser.add_argument("--lambda2", type=int, default=10,help="G2 data loss lambda2")
parser.add_argument("--alpha1", type=int, default=100, help="the parameter alpha1 of L_mf of total G loss")
parser.add_argument("--alpha2", type=int, default=10, help="the parameter alpha2 of L_adv_gan of total G loss")
parser.add_argument("--sample_interval", type=int, default=4, help="batch interval between saving generator samples")
parser.add_argument("--test_interval", type=int, default=4, help="batch interval between test")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="epoch interval between model checkpoints")
args = parser.parse_args()
print(args)

# Create sample, checkpoint directories and logs
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs("%s/images/%s" % (args.output_path, timestamp), exist_ok=True)
os.makedirs("%s/saved_models/%s" % (args.output_path, timestamp), exist_ok=True)
os.makedirs("%s/best_model/%s" % (args.output_path, timestamp), exist_ok=True)
os.makedirs("%s/runs/%s" % (args.log_path, timestamp), exist_ok=True)

# Datasets and DataLoader
org_transform = [
    transforms.CenterCrop(size=args.img_size),
    transforms.Grayscale(num_output_channels=1), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
]
gt_transform = [
    transforms.CenterCrop(size = args.img_size),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)),
]
train_dataset = InfraredImageDataset(args.train_data_path, args.test_data_path, org_transform, gt_transform, mode='train')
test_dataset = InfraredImageDataset(args.train_data_path, args.test_data_path, org_transform, gt_transform, mode='test', test_data='MDvsFA')
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.n_cpus
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.test_batch_size,
    shuffle = True,
    num_workers=args.n_cpus
)

# define model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G1 = Generator1_MD(channel_num=args.channel_num).to(device)
G2 = Generator2_FA(channel_num=args.channel_num).to(device)
D = Discriminator(batch_size=args.batch_size).to(device)
# load model
if args.epoch != 0:
    # load pretrained model
    G1.load_state_dict(torch.load("%s/G1_%d.pth" % (args.pretrain_model_path, args.epoch)))
    G2.load_state_dict(torch.load("%s/G2_%d.pth" % (args.pretrain_model_path, args.epoch)))
    D.load_state_dict(torch.load("%s/D_%d.pth" % (args.pretrain_model_path, args.epoch)))
else:
    # Initialize weights
    G1.apply(weights_init)
    G2.apply(weights_init)
    D.apply(weights_init)

# Optimizers
optimizer_G1 = torch.optim.Adam(G1.parameters(), lr=1e-4, betas=(0.5,0.999))
optimizer_G2 = torch.optim.Adam(G2.parameters(), lr=1e-4, betas=(0.5,0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=1e-5, betas=(0.5,0.999))   

# Losses
adversarial_loss = nn.BCEWithLogitsLoss() # 多分类损失

# ----------
#  Training
# ----------

writer = SummaryWriter("%s/runs/%s"%(args.log_path, timestamp))
best_val_F1 = 0
best_val_g1_F1 = 0
best_val_g2_F1 = 0
for epoch in tqdm(range(args.epoch, args.n_epochs), desc='epoch', position=1):
    # 调整学习率
    if (epoch+1) % 10 == 0:
        for p in optimizer_G1.param_groups:
            p['lr'] *= 0.2
        for q in optimizer_G2.param_groups:
            q['lr'] *= 0.2
        for r in optimizer_D.param_groups:
            r['lr'] *= 0.2 # 
    for i, batch_data in enumerate(tqdm(train_dataloader, desc='batch', position=0, colour='green')):

        torch.cuda.empty_cache() # 释放之前占用的缓存

        # Model input
        org_img = batch_data['org'].to(device)
        gt_img = batch_data['gt'].to(device)

        # Adversarial grount truth
        valid = torch.ones((org_img.shape[0],1),requires_grad=False).to(device)
        fake = torch.zeros((org_img.shape[0],1), requires_grad=False).to(device)
        gt_real = torch.cat((valid,fake,fake),dim=1)
        gt_fake1 = torch.cat((fake, valid, fake), dim=1)
        gt_fake2 = torch.cat((fake, fake, valid),dim=1)

        # ------------------
        #  Train Discriminator
        # ------------------
        D.train()
        G1.eval()
        G2.eval()
        optimizer_G1.zero_grad()
        optimizer_G2.zero_grad()
        optimizer_D.zero_grad()

        g1_out = G1(org_img) # [B,1,128,128]
        g2_out = G2(org_img) # [B,1,128,128]

        # 将G的输出约束到 0-1之间
        g1_out = torch.clamp(g1_out, 0.0, 1.0)
        g2_out = torch.clamp(g2_out, 0.0, 1.0)
        real = torch.cat((org_img, 2*gt_img-1),dim=1) # [B,2,128,128]
        fake_1 = torch.cat((org_img, 2*g1_out-1),dim=1) # [B,2,128,128]
        fake_2 = torch.cat((org_img, 2*g2_out-1),dim=1) # [B,2,128,128]

        d_input = torch.cat((real, fake_1, fake_2), dim=0) # [3B,2,128,128]
        score_real, score_fake1, score_fake2, fakedist = D(d_input) # [B,3]*3, 1

        D_loss_1 = adversarial_loss(score_real, gt_real)
        D_loss_2 = adversarial_loss(score_fake1, gt_fake1)
        D_loss_3 = adversarial_loss(score_fake2, gt_fake2)
        D_loss = D_loss_1 + D_loss_2 + D_loss_3

        D_loss.backward()
        optimizer_D.step()

        # ------------------
        #  Train Generator1
        # ------------------
        D.eval()
        G1.train()
        G2.eval()
        optimizer_G1.zero_grad()
        optimizer_G2.zero_grad()
        optimizer_D.zero_grad()

        g1_out = G1(org_img) # [B,1,128,128]
        g2_out = G2(org_img) # [B,1,128,128]

        # 将G的输出约束到 0-1之间
        g1_out = torch.clamp(g1_out, 0.0, 1.0)
        g2_out = torch.clamp(g2_out, 0.0, 1.0)
        real = torch.cat((org_img, 2*gt_img-1),dim=1) # [B,2,128,128]
        fake_1 = torch.cat((org_img, 2*g1_out-1),dim=1) # [B,2,128,128]
        fake_2 = torch.cat((org_img, 2*g2_out-1),dim=1) # [B,2,128,128]

        # G1 data loss
        MD1 = torch.mean(torch.mul(torch.pow(g1_out-gt_img,2),gt_img)) # 约束检测出更多的1, 减小MD
        FA1 = torch.mean(torch.mul(torch.pow(g1_out-gt_img,2),1-gt_img)) # 约束检测出更多的0, 减小FA
        G1_MFloss = args.lambda1*MD1+FA1

        d_input = torch.cat((real, fake_1, fake_2), dim=0) # [3B,2,128,128]
        score_real, score_fake1, score_fake2, fakedist = D(d_input) # [B,3]*3, 1

        G1_advloss = adversarial_loss(score_fake1, gt_real) # 约束 G1 out 与 gt 相近
        G1_GCloss = fakedist # 约束 G1 out 与 G2 out 相近
        G1_loss = args.alpha1*G1_MFloss+args.alpha2*G1_advloss+G1_GCloss

        G1_loss.backward()
        optimizer_G1.step()

        # ------------------
        #  Train Generator2
        # ------------------
        D.eval()
        G1.eval()
        G2.train()
        optimizer_G1.zero_grad()
        optimizer_G2.zero_grad()
        optimizer_D.zero_grad()

        g1_out = G1(org_img) # [B,1,128,128]
        g2_out = G2(org_img) # [B,1,128,128]

        # 将G的输出约束到 0-1之间
        g1_out = torch.clamp(g1_out, 0.0, 1.0)
        g2_out = torch.clamp(g2_out, 0.0, 1.0)
        real = torch.cat((org_img, 2*gt_img-1),dim=1) # [B,2,128,128]
        fake_1 = torch.cat((org_img, 2*g1_out-1),dim=1) # [B,2,128,128]
        fake_2 = torch.cat((org_img, 2*g2_out-1),dim=1) # [B,2,128,128]

        # G2 data loss
        MD2 = torch.mean(torch.mul(torch.pow(g2_out-gt_img,2),gt_img)) # 约束检测出更多的1, 减小MD
        FA2 = torch.mean(torch.mul(torch.pow(g2_out-gt_img,2),1-gt_img)) # 约束检测出更多的0, 减小FA
        G2_MFloss = MD2+args.lambda2*FA2

        d_input = torch.cat((real, fake_1, fake_2), dim=0) # [3B,2,128,128]
        score_real, score_fake1, score_fake2, fakedist = D(d_input) # [B,3]*3, 1

        G2_advloss = adversarial_loss(score_fake2, gt_real) # 约束 G2 out与 gt 相近
        G2_GCloss = fakedist # 约束 G2 out 与 G1 out 相近
        G2_loss = args.alpha1*G2_MFloss+args.alpha2*G2_advloss+G2_GCloss

        G2_loss.backward()
        optimizer_G2.step()
        
        # log 
        batches_done = epoch*len(train_dataloader)+i
        writer.add_scalars('loss',{
            'D_loss': D_loss,
            "G1_loss": G1_loss,
            'G2_loss': G2_loss
        }, global_step=batches_done)
        # add model graph
        if epoch==0 and i==0:
            # writer.add_graph(G1, org_img)
            writer.add_graph(G2, org_img)
            # writer.add_graph(D, d_input)

        # save model checkpoints
        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
            torch.save(G1.state_dict(), '%s/saved_models/%s/G1_%d.pth' % (args.output_path, timestamp, epoch))
            torch.save(G2.state_dict(), '%s/saved_models/%s/G2_%d.pth' % (args.output_path, timestamp, epoch))
            torch.save(D.state_dict(), '%s/saved_models/%s/D_%d.pth' % (args.output_path, timestamp, epoch))

        # ------------------
        #  Test
        # ------------------
        if batches_done % args.test_interval == 0:
            G1.eval()
            G2.eval()
            D.eval()
            optimizer_G1.zero_grad()
            optimizer_G2.zero_grad()
            optimizer_D.zero_grad()

            sum_g1_F1 = 0
            sum_g2_F1 = 0
            sum_g_F1 = 0

            sum_g1_prec = 0
            sum_g2_prec = 0
            sum_g_prec = 0

            sum_g1_recall = 0
            sum_g2_recall = 0
            sum_g_recall = 0

            for j, batch_data_test in enumerate(test_dataloader):
                if batches_done % args.sample_interval == 0 and j==0:
                    save_path = "%s/images/%s" % (args.output_path, timestamp)
                    sample_images(G1, G2, batches_done, batch_data_test, args.test_batch_size, save_path, device)
                org_img = batch_data_test['org'].to(device)
                gt_img = batch_data_test['gt'].to(device)

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
            
            sum_g1_F1 = sum_g1_F1/len(test_dataloader)
            sum_g2_F1 = sum_g2_F1/len(test_dataloader)
            sum_g_F1 = sum_g_F1/len(test_dataloader)

            sum_g1_prec = sum_g1_prec/len(test_dataloader)
            sum_g2_prec = sum_g2_prec/len(test_dataloader)
            sum_g_prec = sum_g_prec/len(test_dataloader)

            sum_g1_recall = sum_g1_recall/len(test_dataloader)
            sum_g2_recall = sum_g2_recall/len(test_dataloader)
            sum_g_recall = sum_g_recall/len(test_dataloader)

            # log
            writer.add_scalars('test_F1',{
                'g1_F1': sum_g1_F1,
                'g2_F1': sum_g2_F1,
                'g_F1': sum_g_F1
            },global_step=batches_done)

            writer.add_scalars('test_prec',{
                'g1_prec': sum_g1_prec,
                'g2_prec': sum_g2_prec,
                'g_prec': sum_g_prec
            },global_step=batches_done)

            writer.add_scalars('test_recall',{
                'g1_recall': sum_g1_recall,
                'g2_recall': sum_g2_recall,
                'g_recall': sum_g_recall
            },global_step=batches_done)
            
            # save best F1 model
            if sum_g_F1 > best_val_F1:
                best_val_F1 = sum_g_F1
                best_epoch = epoch
                torch.save(
                    G1.state_dict(), "%s/best_model/%s/G1_trade.pth" % (args.output_path, timestamp)
                )
                torch.save(
                    G2.state_dict(), "%s/best_model/%s/G2_trade.pth" % (args.output_path, timestamp)
                )
                torch.save(
                    D.state_dict(), "%s/best_model/%s/D_trade.pth" % (args.output_path, timestamp)
                )
                print('')
                print("best epoch: %d, best F1: %.4f"%(best_epoch, best_val_F1))
                print('')

            if sum_g1_F1 > best_val_g1_F1:
                best_val_g1_F1 = sum_g1_F1
                torch.save(
                    G1.state_dict(), "%s/best_model/%s/G1_single.pth" % (args.output_path, timestamp)
                )
            if sum_g2_F1 > best_val_g2_F1:
                best_val_g2_F1 = sum_g2_F1
                torch.save(
                    G2.state_dict(), "%s/best_model/%s/G2_single.pth" % (args.output_path, timestamp)
                )

writer.close()
print('')
print('')
print("best epoch: %d, best F1: %.4f"%(best_epoch, best_val_F1))





        