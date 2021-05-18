#Explloratory Vision Analysis

from torchvision import datasets,transforms
import os
import sys
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from rf_calc import receptive_field
import matplotlib.pyplot as plt
from skimage import io

import numpy as np

def data_sample():
    demo_transforms = transforms.Compose([transforms.ToTensor()])
    demo_data = datasets.MNIST(root = './',download=True,train = True,transform = demo_transforms)

    data_s = demo_data.test_data
    data_s = demo_data.transform(data_s.numpy())
    return demo_data,data_s

def data_stats(data_s):
    mean = torch.mean(data_s)
    std = torch.std(data_s)
    print (f'The Train Mean is {torch.mean(data_s).item()}')
    print(f'The Train Variance is {torch.var(data_s).item()}')
    print(f'The Train Standard Dev id {torch.std(data_s).item()}')
    print(f'The Train Min is {torch.min(data_s).item()}')
    print (f'The Train Max is {torch.max(data_s).item()}')
    return mean.item(),std.item()

def im_eda(demo_data):
    fig = plt.figure()
    for i in range(1,61):
        a = np.random.randint(0,demo_data.test_data.numpy().shape[0])
        _ = plt.subplot(6,10,i)
        _ = plt.imshow(demo_data.test_data[a].numpy(),aspect='equal')
        _ = plt.axis('off')
    return plt.show()

def gpu_check(seed_val = 1):
    print('The Seed is set to {}'.format(seed_val))
    if torch.cuda.is_available():
        print('Model will Run on CUDA.')
        print ("Type 'watch nvidia-smi' to monitor GPU\n")
        torch.cuda.manual_seed(seed_val)
        device = 'cuda'
    else:
        torch.manual_seed(seed_val)
        print ('Running in CPU')
        device = 'cpu'
    cuda = torch.cuda.is_available()
    return cuda,seed_val,device


def wrong_pred(model,test_data,num_of_image,row,col):
    a = model.eval()
    miss_class  = []
    pred_class = []
    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to('cuda'),target.to('cuda')
            y_test = model(data)
            pred = y_test.argmax(dim = 1,keepdim = True)
            pred_class.extend(list(pred.cpu().numpy()[:,0]))
            op = pred.eq(target.view_as (pred))
            miss_class.extend(list(op.cpu().numpy()[:,0]))

    fig = plt.figure(figsize=(12,12))
    count = 1
    for ids,val in enumerate(miss_class):
        if (val  == False) & (count <= num_of_image):
            _ = fig.add_subplot(row,col,count)
            _ = plt.imshow(test_data.test_data[ids])
            _ = plt.axis('off')
            _ = plt.title(f'Predicted {pred_class[ids]}, Actual {test_data.targets[ids].item()}')
            count += 1
    fig.savefig('Wrong_classified.png')