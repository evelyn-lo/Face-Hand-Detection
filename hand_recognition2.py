import time
import torch
import numpy as np
import os
from PIL import Image
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt

gesture_to_id = {
    "none": 0,
    "paper": 1,
    "rock": 2,
    "scissors": 3
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
        # self.data, self.targets = self.load(self.data_path, train)
        self.data_list = []
        for gesture_name in os.listdir(data_path):
            for img_name in os.listdir(os.path.join(data_path, gesture_name)):
                if train and int(os.path.splitext(img_name)[0]) > 50:
                    self.data_list.append((img_name, gesture_name))
                elif not train and int(os.path.splitext(img_name)[0]) <= 50:
                    self.data_list.append((img_name, gesture_name))
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        img_name, gesture = self.data_list[idx]
        image_path = os.path.join(self.data_path, gesture, img_name)
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, gesture_to_id[gesture]
        
    # def load(self, data_path, train):
    #     images = []
    #     targets = []
    #     for class_name in os.listdir(data_path):
    #         target = class_dict[class_name]
    #         curr_path = os.path.join(data_path, class_name)
    #         for image_name in os.listdir(curr_path):
    #             if 'TR' in image_name and train:
    #                 images.append(os.path.join(curr_path, image_name))
    #                 targets.append(target)
    #             elif 'TE' in image_name and not train:
    #                 images.append(os.path.join(curr_path, image_name))
    #                 targets.append(target)
        
    #     indices = np.random.permutation(len(images))
    #     images = np.array(images)[indices]
    #     targets = np.array(targets, dtype=np.int64)[indices]
    #     return images, targets

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


def run_train():
    transformer = transforms.Compose([
                                    transforms.Grayscale(),
                                    transforms.Resize((128, 128)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()
                                ])

    train_dataset= HandGestureDataset(data_path='rps_data_sample', train=True, transform= transformer)
    test_dataset = HandGestureDataset(data_path='rps_data_sample', train=False, transform= transformer)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    torch.cuda.empty_cache()

    n_classes = 4
    net = Net(n_classes).to(device)


    lr = 0.000000001

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    batch_size = 32
    train_loader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size= batch_size)
    test_loader = torch.utils.data.dataloader.DataLoader(test_dataset, batch_size= batch_size)

    n_epochs = 50
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

if __name__ =="__main__":
    run_train()

