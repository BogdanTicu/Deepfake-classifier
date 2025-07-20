import os
import copy
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt

class ClasificatorDeepFake(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block1=nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        self.conv_block2=nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.conv_block3=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )

        self.conv_block4=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.fc1=nn.Linear(256 * 6 * 6, 512)
        self.dropout_fc=nn.Dropout(0.5)
        self.out=nn.Linear(512, 5)

    def forward(self, x):
        x=self.conv_block1(x)
        x=self.conv_block2(x)
        x=self.conv_block3(x)
        x=self.conv_block4(x)
        x=x.view(x.size(0), -1)
        x=F.relu(self.fc1(x))
        x=self.dropout_fc(x)
        x=self.out(x)
        return x



def filter_frequencies(fft_image):

    h, w=fft_image.shape
    y, x=np.ogrid[:h, :w]
    center_y, center_x=h // 2, w // 2
    distance=np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask=(distance >= 5) & (distance <= 30)
    return fft_image * mask

#we extract fft signals and filter them
#in order to avoid unusual noise
def extract_fft_channels(image_tensor):
    fft_channels=[]
    for i in range(3):
        channel=image_tensor[i].numpy()
        fft=np.fft.fft2(channel)
        fft_shift=np.fft.fftshift(fft)
        filtered_fft=filter_frequencies(fft_shift)
        magnitude=np.log1p(np.abs(filtered_fft))

        magnitude=(magnitude - magnitude.min()) / (
            magnitude.max() - magnitude.min() + 1e-8
        )
        fft_channels.append(torch.tensor(magnitude, dtype=torch.float32))
    return torch.stack(fft_channels)


#we load image in rgb format and then we extract the fft channels and we combine 
#the channels
class ImgLoader(Dataset):
    def __init__(self, df, image_dir, transform=None, include_labels=True):

        self.df=df.reset_index(drop=True)
        self.image_dir=image_dir
        self.transform=transform
        self.include_labels=include_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id=self.df.iloc[idx]['image_id']
        image_path=os.path.join(self.image_dir, image_id + '.png')
        image=Image.open(image_path).convert('RGB')

        image_tensor=TF.to_tensor(image)
        if self.transform:
            image_tensor=self.transform(image_tensor)

        fft_tensor=extract_fft_channels(image_tensor)  
        full_tensor=torch.cat((image_tensor, fft_tensor), dim=0)

        if self.include_labels:
            label=int(self.df.iloc[idx]['label'])
            return full_tensor, torch.tensor(label, dtype=torch.long)
        else:
            return full_tensor


train_data=pd.read_csv('/kaggle/input/kagglecontest/train.csv')
val_data= pd.read_csv('/kaggle/input/kagglecontest/validation.csv')
test_data =pd.read_csv('/kaggle/input/kagglecontest/test.csv')

train_dir='/kaggle/input/kagglecontest/train'
val_dir='/kaggle/input/kagglecontest/validation'
test_dir='/kaggle/input/kagglecontest/test'

transform=transforms.Compose([
    transforms.Resize((100, 100))
])
#we load images
train_dataset=ImgLoader(train_data, train_dir, transform=transform, include_labels=True)
val_dataset=ImgLoader(val_data, val_dir, transform=transform, include_labels=True)
test_dataset=ImgLoader(test_data, test_dir, transform=transform, include_labels=False)

train_loader=DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader=DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
test_loader=DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)



device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
classif=ClasificatorDeepFake().to(device)


weights=compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data['label']),
    y=train_data['label']
)
#we set crossEntropyLoss and optimize with Adam
#we set the scheduler Cosine to adapt learning rate through training
weights_tensor=torch.tensor(weights, dtype=torch.float32).to(device)
nr_epochs=100
criterion=nn.CrossEntropyLoss(weight=weights_tensor)
optimizer=torch.optim.Adam(classif.parameters(), lr=0.001)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nr_epochs, eta_min=1e-5)

best_model_state=None
best_accuracy=0.0

#we train the model for 100 epochs
for epoch in range(nr_epochs):
    classif.train()
    running_loss=0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{nr_epochs}"):
        images, labels=images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs=classif(images)
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    classif.eval()
    val_preds, val_labels=[], []
    with torch.no_grad():
        for images, labels in val_loader:
            images=images.to(device)
            outputs=classif(images)
            preds=torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.numpy())

#we calculate the accuracy score on validation test
    acc_val=accuracy_score(val_labels, val_preds)
    scheduler.step()

    avg_loss=running_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Accuracy={acc_val:.4f}")

#we save the best model with the best accuracy.
    if acc_val > best_accuracy:
        best_accuracy=acc_val
        best_model_state=copy.deepcopy(classif.state_dict())
        print(f" Model saved at: {epoch+1} cu Val Accuracy={acc_val:.4f}")

#we load the best model.
if best_model_state is not None:
    classif.load_state_dict(best_model_state)
    print(f"Best model: ({best_accuracy:.4f}) ")

classif.eval()
predictions=[]

#we do the predictions with the best model and create the csv file
with torch.no_grad():
    for images in tqdm(test_loader, desc="Predict Test"):
        images=images.to(device)
        outputs=classif(images)
        preds=torch.argmax(outputs, dim=1).cpu().numpy()
        predictions.extend(preds)

submission_df=pd.DataFrame({
    'image_id': test_data['image_id'],
    'label': predictions
})
submission_df.to_csv('submission.csv', index=False)
print("Submission saved.")