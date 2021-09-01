import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class Dataset_torch(Dataset):

    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data['y'])    

    def __getitem__(self, idx):
        inputs = [self.data['x1'][idx],self.data['x2'][idx].float()]
        return inputs, self.data['y'][idx]
        
data_1516 = torch.load('./data/tensors_2015_2016.pt')
data_17 = torch.load('./data/tensors_2017.pt')

dataset_1516 = Dataset_torch(data_1516)
data_loader_1516 = torch.utils.data.DataLoader(dataset=dataset_1516, batch_size=256,shuffle=True,num_workers=2)

dataset_2017 = Dataset_torch(data_17)
data_loader_2017 = torch.utils.data.DataLoader(dataset=dataset_2017, batch_size=256,num_workers=2)

data_18 = torch.load('./data/tensors_2018.pt')
dataset_2018 = Dataset_torch(data_18)
data_loader_2018 = torch.utils.data.DataLoader(dataset=dataset_2018, batch_size=1024,num_workers=2)

class Net_spectral(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)

        self.maxpooling2d = nn.MaxPool2d(2, 2)
        self.globalmaxpooling2d = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(p=0.5)

        self.fc256_n1 = nn.Linear(256, 256)
        self.fc256_n2 = nn.Linear(256, 256)

    def forward(self, x):
        x = self.maxpooling2d(F.relu(self.conv1(x)))
        x = self.maxpooling2d(F.relu(self.conv2(x)))
        x = self.maxpooling2d(F.relu(self.conv3(x)))
        x= self.globalmaxpooling2d(x)
        x = F.relu(self.fc256_n1(torch.squeeze(x)))
        x = F.relu(self.fc256_n2(x))
        x = self.dropout(x)
        return x

class Net_cwave(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout(p=0.5)
        
        self.fc10_256 = nn.Linear(10, 256)
        self.fc256_n1 = nn.Linear(256, 256)
        self.fc256_n2 = nn.Linear(256, 256)
        self.fc256_n3 = nn.Linear(256, 256)
        self.fc256_n4 = nn.Linear(256, 256)
        self.fc256_n5 = nn.Linear(256, 256)
        self.fc256_n6 = nn.Linear(256, 256)
        self.fc256_n7 = nn.Linear(256, 256)
        self.fc256_n8 = nn.Linear(256, 256)
        self.fc256_n9 = nn.Linear(256, 256)
        self.fc256_n10 = nn.Linear(256, 256)

    def forward(self, x):
        x = F.relu(self.fc10_256(x[:,22:]))
        x = F.relu(self.fc256_n1(x))
        x = F.relu(self.fc256_n2(x))
        x = F.relu(self.fc256_n3(x))
        x = F.relu(self.fc256_n4(x))
        x = F.relu(self.fc256_n5(x))
        x = F.relu(self.fc256_n6(x))
        x = F.relu(self.fc256_n7(x))
        x = F.relu(self.fc256_n8(x))
        x = F.relu(self.fc256_n9(x))
        x = F.relu(self.fc256_n10(x))
        x = self.dropout(x)
        return x

# Combination of two NN
class Net_comb(pl.LightningModule):
    def __init__(self):
        super().__init__()


        self.maxpooling2d = nn.MaxPool2d(2, 2)
        self.globalmaxpooling2d = nn.AdaptiveMaxPool2d(1)

        self.dropout_n1 = nn.Dropout(p=0.5)
        self.dropout_n2 = nn.Dropout(p=0.5)

        self.fc512_256 = nn.Linear(512, 256)
        self.fc256 = nn.Linear(256, 256)
        self.fc256_2 = nn.Linear(256, 2)

        self.net_cwave = Net_cwave()
        self.net_spectral = Net_spectral()

    def forward(self, x1,x2):

        x = torch.cat((self.net_spectral(x1),self.net_cwave(x2)),1)
        x = F.relu(self.fc512_256(x))
        x = self.dropout_n1(x)
        x = F.relu(self.fc256(x))
        x = self.dropout_n2(x)
        output = F.softplus(self.fc256_2(x))

        return output

    def SE_loss(y_pre,y_true):
        return torch.sum((y_pre - y_true)**2)


    def training_step(self, batch, batch_idx):
        x, y_true = batch
        x1,x2 = x

        x1 = torch.cat([x1, torch.cat([torch.unsqueeze(x1[:,0,:,:],1),torch.unsqueeze(torch.zeros_like(x1[:,1,:,:]),1)],1)],0)
        x2 = torch.cat([x2,x2],0)

        y_pre = self.forward(x1,x2)


        loss_f = nn.GaussianNLLLoss()
        loss = loss_f(y_pre[:,0], torch.cat([torch.squeeze(y_true).float(),torch.squeeze(y_true).float()],0) ,y_pre[:,1])
        self.log('train_loss', loss)

        return loss


    def validation_step(self, batch, batch_idx):

        x, y_true = batch
        x1,x2 = x
        
        y_pre = self.forward(x1,x2)

        loss = F.mse_loss(y_pre[:,0], torch.squeeze(y_true))
        self.log('val_loss', loss)
    
        return loss

    def test_step(self, batch, batch_idx):

        x, y_true = batch
        x1,x2 = x
        
        y_pre = self.forward(x1,x2)
        loss = F.mse_loss(y_pre[:,0], torch.squeeze(y_true))
        self.log('test_loss', loss)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience= 1)
        return { 'optimizer': optimizer, 'lr_scheduler': { 'scheduler': scheduler, 'monitor': 'val_loss', } }

from pytorch_lightning.callbacks import ModelCheckpoint


# Save best model
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./model/models_orignal_mean_var_batch_256_mp_10features_aug',
    filename='model-DNN-p10-{epoch:02d}-{val_loss:.4f}',
    save_top_k=1,
    mode='min',
)

early_stop_callback = EarlyStopping(
monitor='val_loss',
min_delta=0.00,
patience=10,
verbose=True
)

from pytorch_lightning.callbacks import LearningRateMonitor
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# init model
net_comb = Net_comb()

# Initialize a trainer
trainer = pl.Trainer(gpus=[1], max_epochs=100, progress_bar_refresh_rate=200,callbacks=[early_stop_callback,lr_monitor,checkpoint_callback])

# Train the model 
trainer.fit(net_comb, data_loader_1516,data_loader_2017)
trainer.test(test_dataloaders = data_loader_2018)
    
    
