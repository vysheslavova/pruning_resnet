import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import wandb

cuda_num = 0
device = f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu'


def test(model, test_data, criterion):
    model.eval()
    accuracy = []
    losses = []

    with torch.no_grad():
        for image, target in test_data:
            image, target = image.to(device), target.to(device)
            pred = model(image)
            loss = criterion(pred, target)

            losses.append(loss.item())
            accuracy.append(accuracy_score(target.cpu(), torch.argmax(pred, dim=1).cpu()))

    return torch.tensor(accuracy).mean(), torch.tensor(losses).mean()


def train(model, optim, scheduler, criterion, train_data, test_data, epochs, save_path='', log=False):
    best_accuracy = -1
    train_accuracy = []
    for epoch in tqdm(range(epochs), desc='Epoch'):
        model.train()
        losses = []
        for image, target in train_data:
            optim.zero_grad()
            image, target = image.to(device), target.to(device)
            pred = model(image)
            loss = criterion(pred, target)
            loss.backward()
            optim.step()
            losses.append(loss.item())
            train_accuracy.append(accuracy_score(target.cpu(), torch.argmax(pred, dim=1).cpu()))

        test_accuracy, test_loss = test(model, test_data, criterion)

        if save_path and test_accuracy >= best_accuracy:
            torch.save(model.state_dict(), f'weights/{save_path}.pth')
            best_accuracy = test_accuracy
        scheduler.step()
        if log:
            wandb.log({"epoch": epoch, "train_loss": torch.tensor(losses).mean(), "val_loss": test_loss,
                       "train_accuracy": torch.tensor(train_accuracy).mean(), "val_accuracy": test_accuracy})