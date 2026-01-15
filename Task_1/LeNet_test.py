from LeNet import *
model.load_state_dict(torch.load("model.pth"))
model.to(device)
model.eval()
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
def test_(model, testloader, criterion):
        size = len(testloader.dataset)
        num_batches = len(testloader)
        model.eval()
        
        
        class_correct = [0.0 for _ in range(10)]
        class_total = [0.0 for _ in range(10)]
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for X, y in testloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += criterion(pred, y).item()
                pred_classes = pred.argmax(1)
                
            
                correct += (pred_classes == y).type(torch.float).sum().item()
                
                
                for label, pred_class in zip(y, pred_classes):
                    label = label.item()
                    pred_class = pred_class.item()
                    class_total[label] += 1
                    if label == pred_class:
                        class_correct[label] += 1

        
        test_loss /= num_batches
        overall_accuracy = 100 * (correct / size)
        print(f"Test Error: \n Overall Accuracy: {overall_accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")

    
        class_accuracies = {}
        print("Class-wise Accuracy:")
        for i in range(10):
            acc = 100 * (class_correct[i] / class_total[i]) if class_total[i] > 0 else 0
            class_accuracies[CIFAR10_CLASSES[i]] = acc
            print(f"  {CIFAR10_CLASSES[i]}: {acc:>0.1f}% ({int(class_correct[i])}/{int(class_total[i])})")

if __name__ == "__main__":
    test_(model, testloader, criterion)