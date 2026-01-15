import torch 
import torchvision
from torchvision import transforms
from torch import nn
import torch.nn.functional as F 
import time
import os

batch_size = 200
epochs= 20
lr=0.001

# Auto detect GPU and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    gpu_num = torch.cuda.device_count()
    print(f"CUDA is available, Number of GPUs: {gpu_num}")
    for i in range(gpu_num):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA Unavailable, Using CPU")

transform=torchvision.transforms.Compose([
        transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Original DataLoader, shuffle=True restore
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. define multiple convolution and downsampling layers
        # 2. define full-connected layer to classify
        self.cnn = nn.Sequential()
        def cnn_block(in_channels,out_channels):
            return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
            ) if out_channels<256 else nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )
        pre_channels = 3
        for i in range(5):
            out_channels = min(2**(i+6),512)
            self.cnn.add_module(f'cnn_block_{i}', cnn_block(pre_channels, out_channels))
            pre_channels = out_channels
        self.fcnn=nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128,10)
            )
    def forward(self, x: torch.Tensor):
    # x: input image, shape: [B * C * H* W]
    # extract features
    # classification
        out1 = self.cnn(x)
        out1 = out1.flatten(start_dim=1)
        out = self.fcnn(out1)
        return out

def train_loop(model,trainloader,optimizer,criterion,epoch, scheduler):
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

if __name__ == "__main__":
    # Init model + DataParallel for multi-GPU
    model = VGG().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel training")

    # Loss + Optimizer + Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # Total train time
    total_start_time = time.time()

    # Train loop
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        epoch_start = time.time()
        
        train_loop(model, trainloader, optimizer, criterion, t+1, scheduler)        
        epoch_end = time.time()
        epoch_cost = epoch_end - epoch_start
        print(f"Epoch {t+1} Train Time Cost: {epoch_cost//60:.0f}m {epoch_cost%60:.2f}s")
        test_loop(model, testloader, criterion,t+1)
        

        torch.save(model.state_dict(), "model.pth")

    # Total time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"\nTotal Training Time: {total_time//60:.0f}m {total_time%60:.2f}s")
    print("Finish!")