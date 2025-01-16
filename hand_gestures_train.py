import time
import torch
import numpy as np
import os
from PIL import Image
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt

gesture_to_id = {
    ### How we will translate names of hand gestures to numbers
}

class HandGestureDataset(torch.utils.data.dataset.Dataset):
    """
        data_path : path to the folder containing images
        train : to specifiy to load training or testing data 
        transform : Pytorch transforms [required - ToTensor(), optional - rotate, flip]
    """
    def __init__(self, data_path, train = True, transform = None):
        
        self.data_path = data_path
        self.train = train
        # Make a list of tuples of the form (image_name, hand_gesture)
        self.data_list = []
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        image_name, gesture = self.data_list[idx]
        image_path = os.path.join(self.data_path, gesture, image_name)
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, gesture_to_id[gesture]

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        
        self.features = nn.Sequential(
                                nn.Conv2d(1, 6, 3), # in_channels = 1 because we are using grayscale images
                                nn.BatchNorm2d(6, affine = False),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(3,2),
                                nn.Conv2d(6, 12, 3),
                                nn.BatchNorm2d(12, affine = False),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(3,2),
                                nn.Conv2d(12, 24, 3),
                                nn.BatchNorm2d(24, affine = False),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(24, 48, 3),
                                nn.BatchNorm2d(48, affine = False),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(3,2),
                                nn.Conv2d(48, 96, 3),
                                nn.BatchNorm2d(96, affine = False),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(3,2),
        )
        self.classifier = nn.Sequential(
                                nn.Linear(96*4*4,1000),
                                nn.Tanh(),
                                nn.Dropout(p=0.4),
                                nn.Linear(1000, num_classes),
                                nn.Tanh()
        )
                                
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 96*4*4)
        x = self.classifier(x)
        return x
    

def get_accuracy(output, target):
    predictions = torch.argmax(output.data, 1)
    accuracy = (predictions == target).sum().item() / target.size(0)
    return accuracy

def validate(model, device, loader, loss_criterion):
    model.eval()
    losses = []
    accuracies = []
    for idx, (image, target) in enumerate(loader):
        
        image, target = image.to(device), target.to(device)
        
        out = model(image)
        
        loss = loss_criterion(out, target)
        losses.append(loss.item())
        
        accuracy = get_accuracy(out, target)
        accuracies.append(accuracy)
    
    return np.mean(losses), np.mean(accuracies)


def execute_trainstep(model, device, loader, loss_criterion, optimizer):
    model.train()
    losses = []
    accuracies = []
    for idx, (image, target) in enumerate(loader):
        
        image, target = image.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        out = model(image)
        loss = loss_criterion(out, target)
        losses.append(loss.item())
        
        accuracy = get_accuracy(out, target)
        accuracies.append(accuracy)
        
        loss.backward()
        optimizer.step()
        
    return np.mean(losses), np.mean(accuracies)

def train(epochs, model, device, train_loader, valid_loader, loss_criterion, optimizer):
    train_losses = []
    valid_losses = []
    
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(1, epochs+1):
        
        start = time.time()
        
        train_loss, train_accuracy = execute_trainstep(model, device, train_loader, loss_criterion, optimizer)
        valid_loss, valid_accuracy = validate(model, device, valid_loader, loss_criterion)
        
        end = time.time()
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        print(f"""\nEpoch {epoch}/{epochs} Time : {end-start:.2f}s 
                Training Loss : {train_losses[-1]:.6f} Validation Loss : {valid_losses[-1]:.6f}
                Training Accuracy : {train_accuracies[-1]*100:.2f} Validation Accuracy : {valid_accuracies[-1]*100:.2f}""")
        
    return train_losses, valid_losses, train_accuracies, valid_accuracies


transformer = transforms.Compose([
                                transforms.Grayscale(),
                                transforms.Resize((128, 128)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                            ])

# Create datasets for training and testing
train_dataset = None
test_dataset = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

torch.cuda.empty_cache()

# How many different gestures we will be recognizing
n_classes = 0
net = Net(n_classes).to(device)

lr = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

batch_size = 32
train_loader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size= batch_size)
test_loader = torch.utils.data.dataloader.DataLoader(test_dataset, batch_size= batch_size)

# How many times we want to go over the dataset
n_epochs = 20
train_losses, valid_losses, train_accuracies, valid_accuracies = train(n_epochs, net, device, 
                                                                    train_loader, test_loader, criterion, optimizer)

# loss
plt.plot(train_losses, label ='Train')
plt.plot(valid_losses, label ='Valid')
plt.title("Train vs Validation Loss")
plt.legend()
plt.show()

# Accuracy
plt.plot(train_accuracies, label ='Train')
plt.plot(valid_accuracies, label ='Valid')
plt.title("Train vs Validation Accuracy")
plt.legend()
plt.show()

torch.save(net.state_dict(), './bn_hand_gesture_model_'+str(n_epochs)+'.pt')
