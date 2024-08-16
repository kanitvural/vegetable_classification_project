import os
import gradio as gr
import torch
from torchvision import transforms
from model import VegetableModel

model_path = "model/vegetable_epoch_18.pth"
train_folder = "./data/train"
labels = [folder_name for folder_name in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, folder_name))]

print(labels)

num_classes = len(labels)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Preprocess
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# Load Model
model = VegetableModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Prediction function
def predict(image):
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(img_tensor)
    prediction = torch.nn.functional.softmax(prediction, dim=1).squeeze()
    confidences = {labels[i]: float(prediction[i]) for i in range(num_classes)}
    return confidences

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=num_classes),
).launch()



# http://127.0.0.1:7860 