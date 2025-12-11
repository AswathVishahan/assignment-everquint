import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 1 # Reduced for assignment speed (usually 20+)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
DATA_DIR = './data'

# --- 1. MODEL ARCHITECTURE ---
# Requirement: Small CNN from scratch + ResNet Block (Option A)

class ResidualBlock(nn.Module):
    """
    A simple residual block:
    x -> Conv -> BN -> ReLU -> Conv -> BN -> + x -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # If input and output channels differ, we need a 1x1 conv to match dimensions for addition
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Initial Conv Layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2) # 32x32 -> 16x16
        
        # Improvement: Option A - Residual Block
        self.res_block1 = ResidualBlock(32, 64) # 16x16 -> 16x16
        self.pool2 = nn.MaxPool2d(2, 2) # 16x16 -> 8x8
        
        self.res_block2 = ResidualBlock(64, 128) # 8x8 -> 8x8
        self.pool3 = nn.MaxPool2d(2, 2) # 8x8 -> 4x4
        
        # Final Classification Head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.res_block1(x))
        x = self.pool3(self.res_block2(x))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# --- 2. DATA PIPELINE ---
def get_dataloaders():
    print("Preparing Data...")
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(), # Augmentation
        transforms.RandomCrop(32, padding=4), # Augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalize
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)
    
    return trainloader, testloader

# --- 3. TRAINING LOOP ---
def train_model(net, trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    loss_history = []
    
    print(f"Starting Training on {DEVICE}...")
    net.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(trainloader)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f}")
        scheduler.step()
        
    print("Training Finished.")
    torch.save(net.state_dict(), './cifar_net.pth')
    print("Model saved to ./cifar_net.pth")
    return loss_history

# --- 4. EVALUATION ---
def evaluate_model(net, testloader):
    net.eval()
    correct = 0
    total = 0
    misclassified_imgs = []
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Capture some misclassified images
            if len(misclassified_imgs) < 5:
                idxs = (predicted != labels).nonzero()
                for idx in idxs:
                    if len(misclassified_imgs) < 5:
                        img_idx = idx.item()
                        misclassified_imgs.append({
                            'img': images[img_idx].cpu(),
                            'pred': CLASSES[predicted[img_idx]],
                            'true': CLASSES[labels[img_idx]]
                        })

    acc = 100 * correct / total
    print(f"Accuracy on Test Set: {acc:.2f}%")
    return acc, misclassified_imgs

def plot_results(loss_history, mis_imgs):
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    print("Loss plot saved to training_loss.png")
    
    # Show Misclassified
    if mis_imgs:
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i, item in enumerate(mis_imgs):
            img = item['img'] / 2 + 0.5     # unnormalize
            npimg = img.numpy()
            axes[i].imshow(np.transpose(npimg, (1, 2, 0)))
            axes[i].set_title(f"P: {item['pred']} | T: {item['true']}")
            axes[i].axis('off')
        plt.suptitle("Misclassified Examples")
        plt.savefig('misclassified.png')
        print("Misclassified examples saved to misclassified.png")

if __name__ == "__main__":
    trainloader, testloader = get_dataloaders()
    net = Net().to(DEVICE)
    loss_history = train_model(net, trainloader)
    acc, mis_imgs = evaluate_model(net, testloader)
    plot_results(loss_history, mis_imgs)
