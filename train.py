import os
import math
import logging
from typing import List, Dict

from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import CharErrorRate

from dataset import OCRDataset
from model import OCREncoder, OCRDecoder, OCRModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trainer")

writer = SummaryWriter()


def train(epoch, net, dataloader, criterion, optimizer, device="cuda"):
    net.train()
    net = net.to(device)
    
    mean_loss = []
    for k, (img, target) in enumerate(dataloader):
        img = img.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = net(img)
        loss = criterion(output.view(-1, output.size(2)), target.view(-1))
        loss.backward()
        
        nn.utils.clip_grad_norm_(net.parameters(), 1e-1)
        optimizer.step()
        
        mean_loss.append(loss.item())
        writer.add_scalar('training loss', loss.item(), epoch * len(dataloader) + k)

    mean_loss = sum(mean_loss)/len(mean_loss)
    logger.info("train loss: {}".format(round(mean_loss, 3)))
    writer.add_scalar('Mean train loss', mean_loss, epoch)
    return mean_loss


def val(epoch, net, dataloader, criterion, metric, device="cuda"):
    net.eval()
    net = net.to(device)
    
    mean_loss = []
    mean_metric = []
    for img, target in dataloader:
        img = img.to(device)
        target = target.to(device)
        
        output = net(img)
        loss = criterion(output.view(-1, output.size(2)), target.view(-1))
        
        mean_loss.append(loss.item())
        
        preds = []
        real = []
        for i in range(output.size(0)):
            p = output[i].argmax(dim=-1).tolist()
            t = target[i].tolist()
            preds.append(p)
            real.append(t)
        
        mean_metric.append(metric(preds, real).item())
    
    mean_metric = sum(mean_metric)/len(mean_metric)
    mean_loss = sum(mean_loss)/len(mean_loss)
    
    logger.info("CER test: {}".format(round(mean_metric, 3)))
    logger.info("test loss: {}".format(round(mean_loss, 3)))
    writer.add_scalar('Mean test loss', mean_loss, epoch)
    writer.add_scalar('CER test', mean_metric, epoch)

    return mean_loss, mean_metric


transform = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.Resize((48, 80)),
    transforms.ToTensor(),
])

if __name__ == "__main__":
    logger.info("LoadData")
    data = OCRDataset("data/samples", transform=transform)

    train_data, test_data = torch.utils.data.random_split(data, 
                                                      [int(0.8 * len(data)), int(0.2 * len(data))], 
                                                      generator=torch.Generator().manual_seed(42)
                                                     )

    test_data.train_mode = False

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True)

    logger.info("Init net")
    encoder = OCREncoder()
    decoder = OCRDecoder(output_dim=len(data.vocab))
    net = OCRModel(encoder, decoder)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70, 80])
    cer = CharErrorRate()

    best_metric = 1.1

    logger.info("Start train")
    for epoch in range(100):
        logger.info("Epoch {}".format(epoch))
        train(epoch, net, train_loader, criterion, optimizer)
        _, current_metric = val(epoch, net, test_loader, criterion, cer)
        scheduler.step()
        if current_metric < best_metric:
            best_metric = current_metric
            logger.info("Save weights. CER = {}".format(round(best_metric, 3)))
            torch.save(net.state_dict(), "best_model.pth")