# Importy
import csv # Umożliwia pracę z plikami w formacie CSV.
import os # Interakcja z systemem plików
# PyTorch i narzędzia pomocnicze
import torch  # Główna biblioteka do obliczeń tensorowych i sieci neuronowych.
import torchvision  # Narzędzia do przetwarzania obrazów i wczytywania danych.
from torchvision import transforms  # Przekształcenia do preprocessingu obrazów.
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler  # Ładowanie danych i abstrakcja zestawu danych.
import torch.nn as nn  # Moduły sieci neuronowych (warstwy, funkcje aktywacji).
import torch.optim as optim  # Optymalizatory do aktualizacji wag sieci.
from PIL import Image  # Biblioteka do ładowania i manipulowania obrazami.
from tqdm import tqdm  # Pasek postępu do wizualizacji postępu treningu.
from sklearn.model_selection import KFold # Określa liczbę podziałów, które chcemy wykonać na danych.


# Definicja modelu sieci neuronowej
class SimpleCNN(nn.Module):

    # Konstruktor klasy z warstwami sieci, w tym warstwami konwolucyjnymi i liniowymi.
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Inicjalizacja klasy bazowej, konfiguracja warstw.
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    # Metoda definiująca przepływ danych przez sieć (forward pass).
    def forward(self, x):
        # Przekształcenia wejściowego tensora obrazu przez kolejne warstwy.
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Klasa niestandardowego zestawu danych
class CustomDataset(Dataset):
    # Konstruktor klasy zestawu danych, inicjalizacja ścieżek do obrazów.
    def __init__(self, root, transform=None):
        # Lista plików obrazów w folderze root.
        self.root = root
        self.transform = transform
        self.files = [os.path.join(root, file) for file in os.listdir(root) if file.endswith('.jpg')]
        self.labels = [0] * len(self.files)

    # Metoda zwracająca liczbę obrazów w zestawie danych.
    def __len__(self):
        # Zwraca liczbę obrazów.
        return len(self.files)

    # Metoda zwracająca pojedynczy element danych (obraz i etykietę).
    def __getitem__(self, idx):
        # Ładuje obraz, przekształca go i zwraca.
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Funkcja wczytująca dane
def load_data(train_dir, batch_size=4, img_size=224):
    # Przygotowuje transformacje obrazów, wczytuje zestaw danych i tworzy DataLoader.
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print('Dane zostały wczytane...')
    return train_loader, transform


def predict_image(image_path, model, transform):
    # Przetwarza pojedynczy obraz i wykonuje predykcję za pomocą modelu.
    image = Image.open(image_path)
    image = transform(image).float()
    image = image.unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted



def train_model(model, train_loader, criterion, optimizer, num_epochs=5, early_stopping_patience=5):
    # Pętla treningowa, która aktualizuje wagi modelu przy użyciu danych i optymalizatora.
    model.train()  # Przełączanie modelu w tryb treningowy
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}')

        for i, data in progress_bar:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))

        epoch_loss = running_loss / len(train_loader)
        print(f'\nKoniec epoki {epoch + 1}, średnia strata: {epoch_loss:.3f}')

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stopping_patience:
                print("Early stopping zainicjowane")
                break

    print('Zakończono trening modelu')


# Funkcja testująca bez etykiet
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

# Główny blok kodu
def main():
    print("Rozpoczynanie programu...")

    # Ustawienie ścieżek do katalogów z danymi
    train_dir = './train'
    test_dir = './test'

    # Ładowanie danych treningowych
    train_loader, transform = load_data(train_dir)
    num_classes = len(train_loader.dataset.classes)

    # Inicjalizacja KFold do walidacji krzyżowej
    kfold = KFold(n_splits=5, shuffle=True)
    best_model = None
    best_accuracy = 0

    # Otwarcie pliku do zapisu wyników walidacji
    with open('wyniki_walidacji.csv', 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Fold', 'Accuracy'])

        # Iterowanie przez foldy walidacji krzyżowej
        for fold, (train_ids, val_ids) in enumerate(kfold.split(train_loader.dataset)):
            print(f'Rozpoczynanie treningu dla FOLD {fold}')

            # Inicjalizacja modelu i optymalizatora
            model = SimpleCNN(num_classes)
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            criterion = nn.CrossEntropyLoss()

            # Tworzenie DataLoaderów dla treningu i walidacji
            train_subsampler = SubsetRandomSampler(train_ids)
            val_subsampler = SubsetRandomSampler(val_ids)
            train_fold_loader = DataLoader(train_loader.dataset, batch_size=4, sampler=train_subsampler)
            val_fold_loader = DataLoader(train_loader.dataset, batch_size=4, sampler=val_subsampler)

            # Proces treningu modelu
            print(f'Trening modelu dla FOLD {fold}')
            train_model(model, train_fold_loader, criterion, optimizer, num_epochs=10)
            print(f'Zakończenie treningu dla FOLD {fold}')

            # Testowanie modelu i zapisywanie wyników
            print(f'Testowanie modelu dla FOLD {fold}')
            accuracy = test_model(model, val_fold_loader)
            csv_writer.writerow([fold, accuracy])
            print(f'Dokładność testowania dla FOLD {fold} zapisano do pliku')

            # Sprawdzanie i zapisywanie najlepszego modelu
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                torch.save(model.state_dict(), f'model_fold_{fold}.pth')
                print(f'Zapisano model dla FOLD {fold} jako model_fold_{fold}.pth')

        # Zapisywanie najlepszego modelu
        if best_model is not None:
            torch.save(best_model.state_dict(), 'best_model.pth')
            print(f"Najlepszy model zapisany jako 'best_model.pth' z dokładnością: {best_accuracy:.2%}")

    # Ewaluacja końcowa na zbiorze testowym
    print('Ewaluacja końcowa na całym zbiorze testowym')
    test_dataset = CustomDataset(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    predictions = test_model(best_model, test_loader)
    print(f"Dokładność testowania na danych testowych: {predictions:.2%}")

    # Przewidywanie klasy dla wybranego obrazu
    print('Przewidywanie dla konkretnego obrazu')
    image_path = 'image_0780.jpg'
    predicted_class_idx = predict_image(image_path, best_model, transform)
    predicted_class = train_loader.dataset.classes[predicted_class_idx]
    print("Przewidywany gatunek kwiatu:", predicted_class)

if __name__ == "__main__":
    main()






