import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from copy import deepcopy
import time

from RMResnet import rmresnet
from prune_plain_model import PruneRMModel


def testCIFAR10(model, test_loader, model_name=" ", device='cpu'):
    model.eval()
    correct = 0
    total_labels = 0
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            total_labels += labels.size()[0]
            output = model(inputs)
            _, pred = torch.max(output.data, 1)
            correct += (pred == labels).sum().item()
    print("model: {}\taccuracy: {}%".format(model_name, correct * 100 / total_labels))


def trainCIFAR10(model: torch.nn.Module, device='cpu', epoch: int = 10, save: bool = False, compact: bool = False):
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for cur_epoch in range(epoch):
        model.train()

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                print("epoch: {}\titer: {}\tlr: {}\tloss: {}".format(cur_epoch + 1, i, scheduler.get_lr()[0], loss))
        scheduler.step()

        testCIFAR10(model, test_loader, 'raw_model', device)
        # temp_model = deepcopy(model)
        # temp_model.eval()
        # testCIFAR10(temp_model, test_loader, 'deploy model', device)
        # temp_model._deploy()
        # PruneModelTool = PruneRMModel(temp_model, prune_percentage=0.5, channel_limit=8).eval()
        # compact_model = PruneModelTool.get_compact_model(False)
        # testCIFAR10(PruneModelTool, test_loader, 'prune tool', device)
        # compact_model.eval()
        # testCIFAR10(compact_model, test_loader, 'compact model', device)

    print('train and test finished......')
    model.eval()
    if save:
        torch.save(model, 'raw_model.pth', _use_new_zipfile_serialization=False)

    print('now test deploy model......')

    model._deploy()
    model.eval()
    testCIFAR10(model, test_loader, 'deploy model', device)
    if save:
        torch.save(model, 'deploy_model.pth', _use_new_zipfile_serialization=False)

    if compact:
        print('the model will be compact......')
        time.sleep(5)
        PruneModelTool = PruneRMModel(model, prune_percentage=0.5, channel_limit=8).eval()
        compact_model = PruneModelTool.get_compact_model(True)
        testCIFAR10(PruneModelTool, test_loader, 'prune tool', device)
        compact_model.eval()
        testCIFAR10(compact_model, test_loader, 'compact model', device)
        if save:
            torch.save(compact_model, 'compact_model.pth', _use_new_zipfile_serialization=False)


def main():
    device = 'cuda'
    epoch = 100
    save = True
    compact = True
    model = rmresnet(layers=[3, 3, 3, 3], last_stride=2, head_dim=2048).to(device)
    trainCIFAR10(model, device, epoch=epoch, save=save, compact=compact)


if __name__ == '__main__':
    main()
