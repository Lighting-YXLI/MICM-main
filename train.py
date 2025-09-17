import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision import transforms
from utils import WarmUpLR, evaInfo
from options import args
import os
from torch import optim
from dataset import ic_dataset
from MICM import VideoTextModel
from scipy.stats import kendalltau, pearsonr
import numpy as np
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

from scipy.stats import spearmanr


def train(epoch):
    model.train()
    for batch_index, (image, video,label, promote) in enumerate(trainDataLoader):
        image = image.to(device)
        video = video.to(device)
        label = label.to(device)

        Opmimizer.zero_grad()

        # Forward pass
        score1, score2,alignloss = model(image,video, promote)

        # Compute primary losses
        loss1 = loss_function(score1, label)
        loss2 = loss_function(score2, label)

        # Combine losses
        loss = loss1 * 0.6 + loss2 * 0.4 + 0.1*alignloss
        loss.backward()

        Opmimizer.step()

        if epoch <= args.warm:
            Warmup_scheduler.step()

        if (batch_index + 1) % (len(trainDataLoader) // 3) == 0:
            print(
                'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tloss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss.item(),
                    #srcc_penalty,
                    Opmimizer.param_groups[0]['lr'],
                    epoch=epoch,
                    trained_samples=batch_index * args.batch_size + len(image),
                    total_samples=len(trainDataLoader.dataset)
                )
            )


def evaluation():
    model.eval()
    all_scores = []
    all_labels = []
    for (image, video,label, promote) in testDataLoader:
        image = image.to(device)
        video = video.to(device)
        label = label.to(device)
        with torch.no_grad():
            score1, score2,_ = model(image, video, promote)
            score = 0.6 * score1 + 0.4 * score2
            #score = model(image,promote)
            all_scores += score.tolist()
            all_labels += label.tolist()

    # 计算评估指标
    val_PLCC = pearsonr(all_scores, all_labels)[0]
    val_SROCC = spearmanr(all_scores, all_labels)[0]
    val_RMSE = np.sqrt((np.array(all_scores) - np.array(all_labels)) ** 2).mean()
    val_RMAE = np.abs(np.array(all_scores) - np.array(all_labels)).mean() / np.array(all_labels).mean()
    val_KRCC = kendalltau(all_scores, all_labels)[0]

    print(f"PLCC: {val_PLCC}")
    print(f"SROCC: {val_SROCC}")
    print(f"RMAE: {val_RMAE}")
    print(f"RMSE: {val_RMSE}")
    print(f"KRCC: {val_KRCC}")

    # 保存最优结果
    global best_srocc
    if 'best_srocc' not in globals():
        best_srocc = -1  # 初始化最佳 SROCC
    if val_SROCC > best_srocc:
        best_srocc = val_SROCC
        output_path = os.path.join(args.ck_save_dir, "C:/D/VQA/IC/ablation/MyNet5_IC9600T1.0.txt")
        with open(output_path, "w") as f:
            f.write("Best Evaluation Results:\n")
            f.write(f"PLCC: {val_PLCC}\n")
            f.write(f"SROCC: {val_SROCC}\n")
            f.write(f"RMAE: {val_RMAE}\n")
            f.write(f"RMSE: {val_RMSE}\n")
            f.write(f"KRCC: {val_KRCC}\n")
            f.write("\nScores and Labels:\n")
            for score, label in zip(all_scores, all_labels):
                f.write(f"Score: {score}, Label: {label}\n")
        print(f"New best SROCC achieved: {val_SROCC}. Results saved to {output_path}")

    info = evaInfo(score=all_scores, label=all_labels)
    print(info + '\n')




if __name__ == "__main__":

    trainTransform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testTransform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainTransforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testTransforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainDataset = ic_dataset(
        txt_path="C:/D/VQA/IC/trainnew.txt",
        img_path="C:/D/VQA/IC/images/",
        vid_path = "C:/D/VQA/IC/IC9600C3Dmotionfeatures/",
        transform=trainTransforms
    )

    trainDataLoader = DataLoader(trainDataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=True
                                 )

    testDataset = ic_dataset(
        txt_path="C:/D/VQA/IC/testnew.txt",
        img_path="C:/D/VQA/IC/images/",
        vid_path = "C:/D/VQA/IC/IC9600C3Dmotionfeatures/",
        transform=trainTransforms
    )

    testDataLoader = DataLoader(testDataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=False
                                )
    if not os.path.exists(args.ck_save_dir):
        os.mkdir(args.ck_save_dir)

    model = VideoTextModel()

    device = torch.device("cuda:{}".format(args.gpu_id))
    model.to(device)

    loss_function = nn.L1Loss()

    # optimize
    params = model.parameters()
    Opmimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    Scheduler = optim.lr_scheduler.MultiStepLR(Opmimizer, milestones=args.milestone, gamma=args.lr_decay_rate)
    iter_per_epoch = len(trainDataLoader)
    if args.warm > 0:
        Warmup_scheduler = WarmUpLR(Opmimizer, iter_per_epoch * args.warm)

    # running
    for epoch in range(1, args.epoch + 1):
        train(epoch)
        if epoch > args.warm:
            Scheduler.step(epoch)
        evaluation()
        torch.save(model.state_dict(), os.path.join(args.ck_save_dir, 'ck_{}.pth'.format(epoch)))










