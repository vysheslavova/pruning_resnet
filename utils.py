import matplotlib.pyplot as plt
import wandb
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split


def dataloader(batch_size, download=False):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=download, transform=transform_train)

    val_size = 5000
    train_size = len(dataset) - val_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=download, transform=transform_test)

    print(f'Train: {len(trainset)}, Validation: {len(valset)}, Test: {len(testset)}')

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, valloader, testloader

def graphics(run_path, step):
    api = wandb.Api()
    run = api.run(run_path)
    history = run.history()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history[step], history['val_accuracy'], label='validation')
    ax1.plot(history[step], history['train_accuracy'], label='train')
    ax1.set_title('Accuracy', fontsize=20)
    ax1.set_xlabel(step, fontsize=15)
    ax1.set_ylabel('accuracy', fontsize=15)
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history[step], history['val_loss'], label='validation')
    ax2.plot(history[step], history['train_loss'], label='train')
    ax2.set_title('Loss', fontsize=20)
    ax2.set_xlabel(step, fontsize=15)
    ax2.set_ylabel('loss', fontsize=15)
    ax2.legend()
    ax2.grid(True)
    
    plt.show()

if __name__ == '__main__':
    graphics('vysheslavova/MIL/lad3sl5n')