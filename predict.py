import torch
from torchvision import models, transforms
from PIL import Image

def load_model():
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('model/brain_tumor_cnn.pth', map_location='cpu'))
    model.eval()
    return model

def predict_image(image_path):
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs.data, 1)
    return 'Tumor' if predicted.item() == 1 else 'No Tumor'
