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


def dataset_split(label_path, data):

    directory, filename = os.path.split(label_path)
    name, extension = os.path.splitext(filename)

    train_filename = f"{name}_train{extension}"
    test_filename = f"{name}_test{extension}"

    train_path = os.path.join(directory, train_filename)
    test_path = os.path.join(directory, test_filename)

    train, test = train_test_split(data, test_size=0.2, random_state=42)

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    
    return train_path, test_path


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

def main(label_path, image_dir, output_dir, BATCH_SIZE, EPOCH):
    data = pd.read_csv(label_path)
    train_path, test_path = dataset_split(label_path, data)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_proxy = OproxyDataset(metadata = train_path,
                                root_dir = image_dir,
                                transform=transforms.Compose([RandomRotate(),ToTensor(),Grayscale(prob = 0.1),
                                                            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    test_proxy = OproxyDataset(metadata = test_path, 
                                root_dir = image_dir,
                                transform=transforms.Compose([ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    train_loader = torch.utils.data.DataLoader(train_proxy, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_proxy, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    net = models.resnet18(pretrained = True)
    feature_size = net.fc.in_features
    net.fc = nn.Sequential()
    model = BinMultitask(net, feature_size, 10, 200, ordinal=False)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    best_acc = 0
    epochs = EPOCH
    for epoch in range(epochs):
        train_ordinal(train_loader, model, optimizer, epoch, BATCH_SIZE)
        if (epoch + 1) % 10 == 0:
            acc = test_ordinal(test_loader, model)
            if acc > best_acc:
                print('state_saving...')
                save_checkpoint({'state_dict': model.state_dict()}, output_dir, model, f"checkpoint_context{context}_type{label_type}")
                best_acc = acc
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate labels from images.")
    parser.add_argument("--l-path", type=str)
    parser.add_argument("--i-dir", type=str)
    parser.add_argument("--o-dir", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epoch", type=int)
    args = parser.parse_args()

    main(args.l_path, args.i_dir, args.o_dir, args.batch_size, args.epoch)
