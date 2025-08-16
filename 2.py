
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --- Data ---
mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tfms)
test_ds  = datasets.CIFAR10(root="./data", train=Fal…
[8:52 pm, 13/8/2025] Shoaib Khan✨✨✨✨: 1) Create venv and install deps
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Generate data, train, evaluate
python src/generate_data.py
python src/train.py
python src/evaluate.py

# 3) Serve the model
uvicorn app.main:app --reload --port 8000
# 4) Test inference
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @app/sample_payload.json