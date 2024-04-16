import tkinter as tk  # Główna biblioteka do tworzenia GUI
from tkinter import filedialog  # Narzędzie do otwierania okna dialogowego wyboru pliku
from PIL import Image, ImageTk  # Praca z obrazami i ich prezentacja w Tkinter
import torch  # Biblioteka do obliczeń tensorowych i sieci neuronowych
from torchvision import transforms  # Narzędzia do przetwarzania obrazów
import torch.nn as nn  # Komponenty sieci neuronowych
import tensorflow as tf



# Słownik mapujący indeksy klas na ich nazwy dla lepszej czytelności wyników
class_names = {
    0: "Stokrotka",
    1: "Mniszek",
    2: "Tulipan",
    3: "Słonecznik",
    4: "Róża"
}

class SimpleCNN(nn.Module):
    # Konstruktor klasy modelu z warstwami konwolucyjnymi i liniowymi
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    # Metoda forward definiuje jak dane przepływają przez sieć
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Funkcja do wczytywania modelu
def load_model():
    # Wczytuje wytrenowany model sieci neuronowej z dysku
    model = SimpleCNN(num_classes=5)  # Liczba klas
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Funkcja do przewidywania klasy obrazu
def predict(model, image):
    # Przetwarza obraz i dokonuje predykcji klasy za pomocą modelu
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)

    # Wykonanie przewidywania
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_index = predicted.item()

    # Pobranie nazwy klasy na podstawie indeksu
    predicted_name = class_names.get(predicted_index, "Nieznana klasa")
    return predicted_index, predicted_name


# Funkcja wywoływana po naciśnięciu przycisku 'Klasyfikuj, wyświetla wynik klasyfikacji
def classify_image():
    global image_path
    if image_path:
        image = Image.open(image_path)
        predicted_index, predicted_name = predict(model, image)
        label_prediction.config(text=f"Klasa: {predicted_index}, Nazwa: {predicted_name}")


# Funkcja do wczytywania obrazu
def upload_image():
    # Funkcja pozwalająca użytkownikowi na wybranie obrazu do klasyfikacji
    global image_path
    file_path = filedialog.askopenfilename()
    image_path = file_path
    img = Image.open(file_path)
    img = img.resize((250, 250), Image.Resampling.LANCZOS)  # Zmieniono tutaj
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img
    label_prediction.config(text="")

model = load_model()  # Wczytanie modelu
image_path = None  # Zmienna do przechowywania ścieżki wybranego obrazu

# Ustawienia okna aplikacji
root = tk.Tk()
root.title("Klasyfikator Kwiatów")

# Inicjalizacja widżetów (etkiet, przycisków) i ich rozmieszczenie
panel = tk.Label(root, text="Wybierz obraz")
panel.pack()

button_upload = tk.Button(root, text="Wczytaj obraz", command=upload_image)
button_upload.pack()

button_classify = tk.Button(root, text="Klasyfikuj", command=classify_image)
button_classify.pack()

label_prediction = tk.Label(root, text="")
label_prediction.pack()

root.mainloop() # Główna pętla aplikacji, uruchamia interfejs użytkownika
# Pętla główna Tkinter, która utrzymuje otwarte okno aplikacji



