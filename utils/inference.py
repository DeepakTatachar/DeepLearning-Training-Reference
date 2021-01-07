"""
@author: Deepak Ravikumar Tatachar, Sangamesh Kodge
@copyright: Nanoelectronics Research Laboratory
"""

import torch

def inference(net, data_loader,device='cpu'):
    net.eval()
    correct = 0
    total = 0
    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(data_loader):
            data = data.to(device)
            labels = labels.to(device)
            out = net(data)
            _, pred = torch.max(out, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size()[0]
        accuracy = float(correct) * 100.0/ float(total)
    return correct, total, accuracy
