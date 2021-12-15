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
        for i in range(30000, 35000):
            k.append(i)
        for i in range(70000, 75000):
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


transformed_dataset = MRIDataset(csv_file, 
                                 top_root_dir, 
                                 transforms.Compose([transforms.ToPILImage(),
                                                     transforms.Resize((192, 192)),
                                                     transforms.ToTensor()]))

print(transformed_dataset[0][0].shape)

val_data = transformed_dataset

batch_size=16
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
            #print(pred.argmax(1))
            loss.append(loss_fn(pred, y).item())
            accuracy.append((pred.argmax(1) == y).type(torch.float).sum().item() / y.shape[0])

    # Avg loss/acc accross
    loss = np.mean(loss)
    accuracy = np.mean(accuracy)
    return loss, accuracy


#Loss
loss_fn = nn.CrossEntropyLoss()

#Optimizer(SGD)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

checkpoint = torch.load("model_output/model.pth")
model.load_state_dict(checkpoint)

# Do validation
validation_loss, validation_acc = validation(val_data_loader, model, loss_fn)

print(f"Extra validation, loss: {validation_loss:>7f}, accuracy: {validation_acc}")