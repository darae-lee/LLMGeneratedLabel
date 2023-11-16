import os
import torch
import argparse
import torch.nn as nn
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from model.model import *
from model.dataloader import *

from skimage import io, transform
from sklearn.model_selection import train_test_split


def dataset_split(label_type, context, data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    train.to_csv(f"./train_context{context}_type{label_type}.csv", index=False)
    test.to_csv(f"./test_context{context}_type{label_type}.csv", index=False)
    
    return


def train_ordinal(train_loader, model, optimizer, epoch, batch_size):
    model.train()
    count = 0                                                       
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):   
        inputs, targets = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(targets.cuda())
        _, _, logit = model(inputs)
        # Soft Label Cross Entropy Loss
        loss = torch.mean(torch.sum(-targets * torch.log(logit+(1e-15)), 1))
        #loss = torch.mean(torch.sum(-targets * torch.log(logit), 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
        
    total_loss /= count
    print('[Epoch: %d] loss: %.5f' % (epoch + 1, total_loss))

def test_ordinal(test_loader, model):
    model.eval()
    correct = 0
    total = 0
    acc = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(targets.cuda())
            _, _, logit = model(inputs)
            _, predicted = torch.max(logit, 1)
            _, answer =  torch.max(targets, 1)
            total += inputs.size(0)
            correct += (predicted == answer).sum().item()
        acc = (correct / total) * 100.0
        print('Test Acc : %.2f' % (acc))
    
    return acc

def save_checkpoint(state, dirpath, model, arch_name):
    filename = '{}.ckpt'.format(arch_name)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)

def main(label_type, context):

    data = pd.read_csv(f"./label_context{context}_type{label_type}.csv")
    dataset_split(label_type, context, data)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_proxy = OproxyDataset(metadata = f"./train_context{context}_type{label_type}.csv",
                                root_dir = "./dataset",
                                transform=transforms.Compose([RandomRotate(),ToTensor(),Grayscale(prob = 0.1),
                                                            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    test_proxy = OproxyDataset(metadata = f"./test_context{context}_type{label_type}.csv", 
                                root_dir = "./dataset",
                                transform=transforms.Compose([ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    train_loader = torch.utils.data.DataLoader(train_proxy, batch_size=50, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_proxy, batch_size=50, shuffle=False, num_workers=0) # 4->0

    net = models.resnet18(pretrained = True)
    feature_size = net.fc.in_features
    net.fc = nn.Sequential()
    model = BinMultitask(net, feature_size, 10, 200, ordinal=False)

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    best_acc = 0
    epochs = 100
    for epoch in range(epochs):
        train_ordinal(train_loader, model, optimizer, epoch, 50)
        if (epoch + 1) % 10 == 0:
            acc = test_ordinal(test_loader, model)
            if acc > best_acc:
                print('state_saving...')
                save_checkpoint({'state_dict': model.state_dict()}, './save_model', model, f"checkpoint_context{context}_type{label_type}")
                best_acc = acc
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate labels from images.")
    parser.add_argument("--type", type=str, help="Specify the type (0, 1, 2, 3, or 4).")
    parser.add_argument("--context", type=int, help="Specify the context (0, 1, 2 or 3).")
    args = parser.parse_args()

    main(args.type, args.context)
