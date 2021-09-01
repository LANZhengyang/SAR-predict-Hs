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
        
data_1517 = torch.load('./data/tensors_2015_2017.pt')

dataset_1517 = Dataset_torch(data_1517)
data_loader_1517 = torch.utils.data.DataLoader(dataset=dataset_1517, batch_size=256,shuffle=True,num_workers=2)


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
        
        self.fc32_256 = nn.Linear(32, 256)
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
        x = F.relu(self.fc32_256(x))
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
        
        y_pre = self.forward(x1,x2.float())


        loss_f = nn.GaussianNLLLoss()
        loss = loss_f(y_pre[:,0], torch.squeeze(y_true).float(),y_pre[:,1])

        # Logging to TensorBoard by default
        self.log('train_loss', loss )

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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6, verbose=True)
        return [optimizer], [scheduler]


from pytorch_lightning.callbacks import ModelCheckpoint


# init model
net_comb = Net_comb()

from pytorch_lightning.callbacks import LearningRateMonitor
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Initialize a trainer
trainer = pl.Trainer(gpus=[2], max_epochs=30, progress_bar_refresh_rate=200,callbacks=[lr_monitor], default_root_dir='./model/model_mean_var_batch_256_fix_epoch_m20/')

# Train the model 
trainer.fit(net_comb, data_loader_1517)
trainer.test(test_dataloaders = data_loader_2018)
trainer.test(test_dataloaders = data_loader_1517)    
    