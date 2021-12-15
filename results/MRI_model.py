import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import os.path as osp
import pandas as pd
import pydicom as dicom
import time

class MRIDataset(Dataset):
    def __init__(self, annotations_file, root_dir, transform=None, target_transform=None):
        self.img_df = pd.read_csv(annotations_file)
        
        ### limiting the size for now
        k = []
        for i in range(20000,20500):
            k.append(i)
        for i in range(50000, 50500):
            k.append(i)
        self.img_df = self.img_df.iloc[k]
        ######
        
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.unique_labels = ['MCI', 'CN', 'EMCI', 'AD', 'LMCI', 'SMC', 'Patient']

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        dcm_filename = self.img_df.iloc[idx]['file']
        label = self.img_df.iloc[idx]['label']
        label = self.unique_labels.index(label)
        
        subject_id = self.img_df.iloc[idx]['subject_id']
        
        ds = dicom.dcmread(dcm_filename)
        image = np.array(ds.pixel_array)
        image = image.astype('float32')
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    
top_root_dir = r"/scratch2/linjohnd/project_data"
csv_filename = 'files_and_labels.csv'

csv_file = osp.join(top_root_dir, csv_filename)
img_dataset = MRIDataset(csv_file, top_root_dir)
print(f"Number of samples: {len(img_dataset)}")

img_df = pd.read_csv(osp.join(top_root_dir, csv_filename))
print("Input classes summary")
print(img_df['label'].value_counts())

class ToTensor(object):
    def __call__(self, image):
        image = torch.from_numpy(image)
        image = torch.unsqueeze(image, 0)

        return image
    
transformed_dataset = MRIDataset(csv_file, 
                                 top_root_dir, 
                                 transforms.Compose([transforms.ToPILImage(),
                                                     #transforms.RandomRotation(37),
                                                     transforms.RandomHorizontalFlip(0.5),
                                                     #transforms.RandomVerticalFlip(0.5),
                                                     transforms.RandomResizedCrop(192, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
                                                     transforms.Resize((192, 192)),
                                                     transforms.ToTensor()]))

print(transformed_dataset[0][0].shape)

# define the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

#print(len(transformed_dataset))

# calculate the train/validation split
num_train_samples = int(len(transformed_dataset) * TRAIN_SPLIT)
#print(num_train_samples)
num_val_samples = int(len(transformed_dataset) * VAL_SPLIT)
#print(num_val_samples)

(train_data, val_data) = random_split(transformed_dataset,
                                    [num_train_samples, num_val_samples],
                                    generator=torch.Generator().manual_seed(42))


batch_size=16
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
val_data_loader = DataLoader(val_data, batch_size=batch_size, num_workers=0)

#Instantiating CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Verifying CUDA
#print(device)
#print(torch.cuda.device_count())

class CNN(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
model = CNN(num_classes = 7, dropout = 0.7)
# lower performance with parallel gpu
#if torch.cuda.device_count() > 1:
#    print(f"GPU count: {torch.cuda.device_count()}")
#    model = nn.DataParallel(model)
print(device)
model.to(device)

    
# Aux function that runs validation and returns val loss/acc
def validation(dataloader, model, loss_fn):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval() # Set the model on inference mode; dropout
    loss, accuracy = [], []
    with torch.no_grad(): # no_grad() skips the gradient computation; faster
        for i, (X,y) in enumerate(dataloader):
            X, y = X.to(device, dtype=torch.float), y.to(device)
            pred = model(X)
            loss.append(loss_fn(pred, y).item())
            accuracy.append((pred.argmax(1) == y).type(torch.float).sum().item() / y.shape[0] )
            # calculate loss and acc only for the first 20 batches for speed
            if i == 20:
                break

    # Avg loss/acc accross
    loss = np.mean(loss)
    accuracy = np.mean(accuracy)
    return loss, accuracy


def plot(train_loss, val_loss, train_acc, val_acc, title=""): 
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].plot(train_loss)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("loss")

    ax[0].plot(val_loss)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("loss")
    ax[0].legend(["train", "validation"])

    ax[1].plot(train_acc)
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("accuracy")

    ax[1].plot(val_acc)
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("accuracy")
    ax[0].legend(["train", "validation"])
    if title:
        fig.suptitle(title)
        
    plt.savefig('model_output/graphs.jpg', format='jpg')

    
#Loss
loss_fn = nn.CrossEntropyLoss()

#Optimizer(SGD)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


epochs = 15
size = len(train_data_loader.dataset)

train_loss = []
val_loss = []
train_acc = []
val_acc = []

output_text = []

# timer starts
start = time.perf_counter()

for epoch in range(epochs):
    #i_batch, sample_batched
    for batch, (x,y) in enumerate(train_data_loader):
        model.train()

        # Transfer data to device
        x, y = x.to(device, dtype=torch.float32), y.to(device) 

        # Pass data through model
        y_pred = model(x)

        # Calculate loss 
        loss = loss_fn(y_pred, y)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            # Calculate accuracy 
            acc = (y_pred.argmax(1) == y).type(torch.float)
            acc = torch.mean(acc)

            # Append to lists 
            train_loss.append(loss.item())
            train_acc.append(acc.item())
    
            # Do validation
            validation_loss, validation_acc = validation(val_data_loader, model, loss_fn)
            val_loss.append(validation_loss)
            val_acc.append(validation_acc)
    
    # Log some info
    loss, current = loss.item(), batch * len(x)
    output_text.append(f"Epoch = {epoch}, train loss: {loss:>7f}, train accuracy: {acc}")
    print(f"Epoch = {epoch}, train loss: {loss:>7f}, train accuracy: {acc}")
    
# timer ends
end = time.perf_counter()
elapsed_time = end - start
output_text.append(f"Elapsed time: {elapsed_time}")
print(f"Elapsed time: {elapsed_time}")
    
    
# Plot the train/val loss and accuracy unnormalized data
plot(train_loss, val_loss, train_acc, val_acc, "")

# Save the model
torch.save(model.state_dict(), 'model_output/model.pth')

# Write to a file
with open("model_output/output.txt", "a") as f:
    for i in output_text:
        f.write(i + "\n")