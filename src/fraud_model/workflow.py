from torchvision import models
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch import nn
from torchvision import transforms
from .crop_fields import CropFields
from .freuid_dataset import FreuidDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from .freuid_submition import FreuidSubmition

class Workflow:
    device: str = 'cuda'
    _bacth_size: int = 128
    _epochs: int = 25
    _workers: int = 4
    _lr: float = 1e-3

    def __init__(self, dataset_path: str, dataset_size: float = 1.0,  batch_size: int = 32,  num_workers=4):
        self._workers = num_workers
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._bacth_size = batch_size
        self._epochs = 25
        self._lr = 1e-4
        self._dataset_path = dataset_path
        self._dataset_size = dataset_size        

    def get_train_labels(self):
        return pd.read_csv(self._dataset_path + '/train_labels.csv')

    def submition(self, filename):
        submeter = FreuidSubmition(filename)
        submeter.run(
            self._dataset_path + '/public_test/public_test',
            self.state_model,
            self.state_transforms
        )

    def split(self)->tuple:
        df = self.get_train_labels()

        w = df.shape[0]

        w = int(w * self._dataset_size)
        df = df.iloc[:w]

        X_train, X_test, y_train, y_test = train_test_split(
            df[['image_path', 'is_digital', 'type']],
            df['label'],
            random_state=42,
            shuffle=True,
            test_size=0.1
        )

        return (X_train, y_train) , (X_test, y_test)

    def train(self, model):
        train_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.state_transforms = train_transforms
        self.state_model = model

        train_df, test_df = self.split()

        X_train, y_train = train_df
        X_test, y_test = test_df

        train_dataset = FreuidDataset(
            self._dataset_path,
            {
                "paths": X_train['image_path'].values,
                "regions": X_train['type'].values,
                "labels": y_train.values
            },
            train_transforms
        )
        test_dataset = FreuidDataset(
            self._dataset_path,
            {
                "paths": X_test['image_path'].values,
                "regions": X_test['type'].values,
                "labels": y_test.values
            },
            train_transforms
        )

        train_loader = DataLoader(train_dataset, batch_size=self._bacth_size, num_workers=self._workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self._bacth_size,  num_workers=self._workers, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.state_model.parameters(), lr=self._lr)

        print("Start model train")
        for epoch in range(1, self._epochs + 1):
            self.state_model.train()
            train_loss = 0.0

            print(f"Epoch [{epoch}/{self._epochs}]")

            for images, labels in train_loader:
                images = images.to(self._device)
                labels = labels.to(self._device)

                optimizer.zero_grad()

                outputs = self.state_model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            print(f"Loss {train_loss:.6f}")
        

            correct = 0
            total = 0
        print("Model train done")

        print("Starting validation")

        self.state_model.eval()

        y_true = []
        y_pred = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self._device)
                labels = labels.to(self._device)

                outputs = self.state_model(images)

                preds = outputs.argmax(dim=1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total

        print("Validation done!")

        print(f"Accuracy: {accuracy:.4f}")
        

        print(classification_report(y_true, y_pred))

        return self.state_model

    def save(self, model, filename):
        torch.save(model.state_dict(), filename)