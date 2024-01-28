import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch import nn

class TinyVGG(nn.Module):
  def __init__(self,input_shape: int,hidden_units: int,output_shape: int) -> None:
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels = input_shape,
                  out_channels = hidden_units,
                  kernel_size = 3,
                  stride = 1,
                  padding = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels = hidden_units,
                  out_channels = hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.MaxPool2d(kernel_size = 2,
                     stride=2)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(hidden_units,hidden_units,kernel_size = 3,padding = 1),
        nn.ReLU(),
        nn.Conv2d(hidden_units,hidden_units,kernel_size = 3,padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = hidden_units*32*32,
                  out_features = output_shape)
    )
  def forward(self,x: torch.Tensor):
    x = self.conv_block_1(x)
    #print(x.shape)
    x = self.conv_block_2(x)
    #print(x.shape)
    x = self.classifier(x)
    #print(x.shape)
    return x
    #return self.classifier(self.conv_block_2(self.conv_block_1(x)))


class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier App")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.load_model("/media/croxx/vixen/mush/Mushroom_Classification_model.pth")  # Replace with the path to your .pth file

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self.root, text="Select an image:")
        self.label.pack(pady=10)

        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        self.classify_button = tk.Button(self.root, text="Classify Image", command=self.classify_image)
        self.classify_button.pack(pady=10)

        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack(pady=10)

    # Inside the upload_image method
    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select Image File")

        if file_path:
            image = Image.open(file_path)
            image = image.resize((128,128))  # Replace Image.ANTIALIAS with "antialias"
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo
            self.image_path = file_path


    def classify_image(self):
        my_class_names = list(["poisonous","edible"])
        if hasattr(self, 'image_path'):
            image = Image.open(self.image_path)
            image = self.transform(image).unsqueeze(0).to(self.device)
            

            # Replace this part with your PyTorch model prediction logic
            with torch.no_grad():
                self.model.eval()
                output = self.model(image)
                probabilities = F.softmax(output[0], dim=0)
                class_label = torch.argmax(probabilities).item()

                self.result_label.config(text=f"Prediction:  {my_class_names[class_label]} \n Probabilty: {probabilities.max():.4f}")
        else:
            self.result_label.config(text="Please upload an image first.")

    def load_model(self, model_path):
        # Replace "your_model.pth" with the path to your .pth file
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        return model

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
