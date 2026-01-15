import torch
import mytensor as mt
import numpy as np
import torchvision
from torchvision import transforms
from utils.model import Model
from utils.optimizer import Optimizer
from utils.scheduler import Scheduler,StepLR,CosineAnnealingLR
import swanlab

batch_size =50
epochs= 50
lr=0.001

swanlab.init(
    project = "ai_pro_final",
    workspace = "TrailblazerWu",
      config={
    "learning_rate": 0.001,
    "architecture": "LeNet-like",
    "dataset": "CIFAR-10",
    "epochs": 50
  }
)
train_transform=torchvision.transforms.Compose([
    transforms.RandomRotation(degrees=5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

test_transform=torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
dataiter = iter(trainloader)
images, labels = next(dataiter)



model = Model(
    mt.Conv(3,32),
    mt.Relu(),mt.Pool(),
    mt.Conv(32,64),
    mt.Relu(),mt.Pool(),
    mt.Conv(64,100),mt.Relu(),mt.Pool(),
    mt.Conv(100,256),
    mt.Relu(),mt.Pool(),
    mt.Linear(1024,256,mt.Device.GPU),mt.Relu(),
    mt.Linear(256,128,mt.Device.GPU),mt.Relu(),
    mt.Linear(128,10,mt.Device.GPU))
optim = Optimizer(model,lr)
scheduler = StepLR(optim,lr,10,0.9)
criterion = mt.CrossEntropyLoss()

def train_loop(model,trainloader,optimizer,criterion,epoch):
    size = len(trainloader.dataset)
    num_batches = len(trainloader)
    train_loss,correct =0,0
    
    for batch,(X,y) in enumerate(trainloader):
        l  =len(X)
        X = np.array(X,dtype=np.float32)
        y = np.array(y,dtype=np.float32)
        X = mt.Tensor.from_numpy(X,mt.Device.GPU)
        y = mt.Tensor.from_numpy(y,mt.Device.GPU)
        
        
        logits = model.forward(X)
        loss = criterion.forward(logits,y)
        train_loss+=loss

        logits_grad =criterion.backward()
        model.backward(logits_grad)

        
        optimizer.step()

        logits.to_cpu()
        logits_np = logits.numpy()
        y.to_cpu()
        y_np = y.numpy().astype(np.int64)
        correct+= (logits_np.argmax(1) == y_np).astype(np.float32).sum().item()

        


        if batch % 100 == 0:
            current = batch * l
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    
    scheduler.step()
    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    swanlab.log({"Train Accuracy": 100*correct, "Train Avg loss": train_loss, "epoch": epoch})

def test_loop(model,testloader,criterion,epoch):
    size=len(testloader.dataset)
    num_batches=len(testloader)
    test_loss, correct = 0, 0
    for X,y in testloader:
        X = np.array(X,dtype=np.float32)
        y = np.array(y,dtype=np.float32)
        X = mt.Tensor.from_numpy(X,mt.Device.GPU)
        y = mt.Tensor.from_numpy(y,mt.Device.GPU)
        
        logits = model.forward(X)
        test_loss += criterion.forward(logits,y)
        logits.to_cpu()
        logits_np = logits.numpy()
        y.to_cpu()
        y_np = y.numpy().astype(np.int64)
        correct += (logits_np.argmax(1) == y_np).astype(np.float32).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    swanlab.log({"Test Accuracy": 100*correct, "Test Avg loss": test_loss, "epoch": epoch})


        




if __name__ == '__main__':
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(model,trainloader,optim,criterion,epoch)
        
        test_loop(model,testloader,criterion,epoch)
        
    print("Done!")
    swanlab.finish()
