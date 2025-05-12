# %%
import os
import cv2
import time
import re
import torch
import easyocr
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt



# %%
data_path = '/Users/feistella/.cache/kagglehub/datasets/gpiosenka/us-license-plates-image-classification/versions/3/new plates'
class_names = os.listdir(os.path.join(data_path, 'train'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # ensures 3-channel
    transforms.Resize(224),  # rescale short side to 224, preserves aspect ratio
    transforms.Pad((0, 0, 0, 0), padding_mode='edge'),  # optional; useful if needed to square it
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.3))], p=0.1),
    transforms.RandomHorizontalFlip(p=0.1),  # rare but possible
    transforms.RandomRotation(degrees=5),    # preserve text layout
   
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.01, 0.05))  # simulate light occlusion
])

test_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # ensure 3 channels
    transforms.Resize(224),             # keep aspect ratio
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load datasets
data_loaders = {
    'train': DataLoader(datasets.ImageFolder(os.path.join(data_path, 'train'), transform=train_transform), batch_size=32, shuffle=True, pin_memory=True, drop_last=True),
    'test': DataLoader(datasets.ImageFolder(os.path.join(data_path, 'test'), transform=test_transform), batch_size=32, shuffle=False, pin_memory=True)
}

# %%
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
summary(model, input_size=(3, 224, 224))

for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if 'layer3' in name or 'layer4' in name:
        param.requires_grad = True

model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(128, len(class_names))
)
model = model.to(device)

# Prepare training tools
params_to_update = [p for p in model.parameters() if p.requires_grad]
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(params_to_update, lr=3e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=25)

# Training loop
train_losses, test_losses, train_acc, test_acc = [], [], [], []
best_acc = 0

def train_one_epoch(model, loader):
    model.train()
    correct, total = 0, 0
    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_acc.append(100 * correct / total)


def evaluate(model, loader):
    model.eval()
    correct, total, test_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_losses.append(test_loss / len(loader))
    acc = 100 * correct / total
    test_acc.append(acc)
    return acc


# %%
for epoch in range(10): #update 25 
    print(f"\nEpoch {epoch + 1}/25")
    train_one_epoch(model, data_loaders['train'])
    acc = evaluate(model, data_loaders['test'])
    print(f"Test Accuracy: {acc:.2f}%")
    scheduler.step()
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pth')


# Plotting
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Train Acc')
plt.plot(test_acc, label='Test Acc')
plt.legend(); plt.title("Accuracy")
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend(); plt.title("Loss")
plt.show()





# %%
import easyocr
import torch
import torch.nn.functional as F
import numpy as np
import os
import re
from PIL import Image
from tqdm import tqdm
import cv2

# OCR Integration
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

def is_likely_plate(text):
    text = text.strip().upper()
    return bool(re.fullmatch(r'[A-Z0-9]{5,8}', text))  # adjusted to reduce false positives

def preprocess_for_ocr(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 15)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

def tensor_to_pil(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img).convert("RGB")

model.eval()
results = []
index = 0

with torch.no_grad():
    for images, _ in tqdm(data_loaders['test']):
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, dim=1)

        for i in range(images.size(0)):
            img_tensor = images[i].cpu()
            pil_img = tensor_to_pil(img_tensor)
            img_path, _ = data_loaders['test'].dataset.samples[index]
            true_state = os.path.basename(os.path.dirname(img_path))
            image_id = f"{true_state}_{os.path.basename(img_path)}"
            pred_state = class_names[preds[i]]
            state_conf = confs[i].item()

            np_img = np.array(pil_img)
            h, w = np_img.shape[:2]
            cropped = np_img[int(0.25 * h):int(0.75 * h), :]  # tighter crop

            processed_img = preprocess_for_ocr(cropped)
            ocr_result = reader.readtext(processed_img)

            best_match = {"text": "", "conf": 0.0}
            for _, text, conf in ocr_result:
                cleaned = text.replace('O', '0').replace('I', '1').replace('S', '5').upper()
                cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
                if is_likely_plate(cleaned) and conf > best_match["conf"]:
                    best_match["text"] = cleaned
                    best_match["conf"] = conf

            results.append({
                "image_id": image_id,
                "true_state": true_state,
                "predicted_state": pred_state,
                "state_confidence": state_conf,
                "plate_number": best_match["text"],
                "ocr_confidence": best_match["conf"]
            })

            index += 1

# %%
pd.DataFrame(results)
# %%
