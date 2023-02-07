import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


from tqdm import tqdm
from PIL import Image
from typing import List, Dict
from torchmetrics import CharErrorRate


from dataset import OCRDataset
from model import OCREncoder, OCRDecoder, OCRModel


def train(net, dataloader, criterion, optimizer, device="cuda"):
    net.train()
    net = net.to(device)
    
    mean_loss = []
    for img, target in dataloader:
        img = img.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = net(img)
        loss = criterion(output.view(-1, output.size(2)), target.view(-1))
        loss.backward()
        
        nn.utils.clip_grad_norm_(net.parameters(), 1e-1)
        optimizer.step()
        
        mean_loss.append(loss.item())
    
    mean_loss = sum(mean_loss)/len(mean_loss)
    print("train loss:", round(mean_loss, 3))
    return mean_loss


def val(net, dataloader, criterion, metric, device="cuda"):
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
    
    print("test metric:", round(mean_metric, 3))
    print("test loss:", round(mean_loss, 3))
    return mean_loss, mean_metric


transform = transforms.Compose([
    transforms.Resize((48, 80)),
    transforms.ToTensor(),
])

if __name__ == "__main__":
	data = OCRDataset("data/samples", transform=transform)

	train_data, test_data = torch.utils.data.random_split(data, 
                                                      [int(0.8 * len(data)), int(0.2 * len(data))], 
                                                      generator=torch.Generator().manual_seed(42)
                                                     )

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True)

	encoder = OCREncoder()
	decoder = OCRDecoder(output_dim=len(data.vocab))
	net = OCRModel(encoder, decoder)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70, 80])
	cer = CharErrorRate()

	for epoch in range(100):
	    print("Epoch", epoch)
	    train(net, train_loader, criterion, optimizer)
	    val(net, test_loader, criterion, cer)
	    scheduler.step()
	    print("==="*20)