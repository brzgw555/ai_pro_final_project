import torch 
import torchvision
from torchvision import transforms
from torch import nn
from torch import functional as F 
import wandb
import os
import swanlab
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 50
epochs= 40
lr=0.001

transform=torchvision.transforms.Compose([
        transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

dataiter = iter(trainloader)
images, labels = next(dataiter)
swanlab.init(
    project = "ai_pro_final",
    workspace = "TrailblazerWu",
      config={
    "learning_rate": 0.001,
    "architecture": "LeNet",
    "dataset": "CIFAR-10",
    "epochs": 40
  }
)



class LeNet(torch.nn.Module):
        def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, stride=1,padding=1, kernel_size=3)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, stride=1, padding=1, kernel_size=3)
                self.fc1 = nn.Linear(1024, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)
                self.relu = nn.ReLU()

        def forward(self ,x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

model = LeNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
def train_loop(model,trainloader,optimizer,criterion,epoch):
        size=len(trainloader.dataset)
        model.train()
        train_loss , correct =0,0
        num_batches = len(trainloader)
        for batch, (X,y) in enumerate(trainloader):
            X,y=X.to(device),y.to(device)
            pred=model(X)
            loss=criterion(pred,y)
            train_loss+=loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        scheduler.step()
        train_loss /= num_batches
        train_acc = correct / size
        swanlab.log({"Train Accuracy torch": train_acc, "Train Avg loss torch": train_loss, "epoch": epoch})
        print(f"Train Error: \n Accuracy: {(100*train_acc):>0.1f}%, Avg loss: {train_loss:>8f} \n")
                

def test_loop(model,testloader,criterion,epoch):
            size=len(testloader.dataset)
            num_batches=len(testloader)
            model.eval()
            test_loss, correct = 0, 0
            with torch.no_grad():
                for X, y in testloader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                    test_loss += criterion(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                test_loss /= num_batches
                correct /= size
                print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
                swanlab.log({"Test Accuracy torch": 100*correct, "Test Avg loss torch": test_loss, "epoch": epoch})

if __name__ == "__main__":
    for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            epoch_start = time.time()  

            train_loop(model, trainloader, optimizer, criterion, t+1)
            epoch_end = time.time()
            print(f"Epoch {t+1} training time: {epoch_end - epoch_start:.2f} seconds")
            test_loop(model, testloader, criterion,t+1)
            torch.save(model.state_dict(), "model.pth")


    swanlab.finish()
    print("Finish!")



    




