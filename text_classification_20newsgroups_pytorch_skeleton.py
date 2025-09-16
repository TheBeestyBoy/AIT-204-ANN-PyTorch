# text_classification_20newsgroups_pytorch_skeleton.py
# Purpose: TF-IDF + PyTorch MLP for 20 Newsgroups (FULL IMPLEMENTATION)

import os
import random
import numpy as np

# ---- Reproducibility (optional but recommended) ----
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# Load Dataset
# =========================
print(">>> Loading dataset...")
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
print(">>> Dataset loaded.")
X_raw, y = data.data, data.target
num_classes = len(data.target_names)

# =========================
# Preprocess text data
#  - Convert to lowercase, remove punctuation if desired
#  - Tokenization is handled by TfidfVectorizer, but you can add custom steps if needed
# =========================
# No additional preprocessing required beyond what TfidfVectorizer handles

# =========================
# Convert Text Data to Numerical Format
# =========================
vectorizer = TfidfVectorizer(
    max_features=5000,          # Limit to 5000 most frequent words
    lowercase=True,
    stop_words='english',
    strip_accents='unicode',
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
)
print(">>> Vectorizing text...")
X_vec = vectorizer.fit_transform(X_raw)
print(">>> Vectorization done. Shape:", X_vec.shape)
print(">>> Converting to dense array...")
X_vec = X_vec.toarray()         # NOTE: densifies; OK at 5k features, but watch RAM on small machines
print(">>> Dense array created.")

# =========================
# Split data
# =========================
print(">>> Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, stratify=y, random_state=SEED
)
print(">>> Split done. Training size:", X_train.shape, "Test size:", X_test.shape)


# =========================
# Torch Tensors & Dataloaders
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

from torch.utils.data import TensorDataset, DataLoader

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

# Batch size tuned for memory constraints
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, drop_last=False)

# =========================
# Design Neural Network Architecture
# =========================
import torch.nn as nn

class NewsMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Define layers with dropout for regularization
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Forward pass using defined layers
        return self.net(x)

input_dim = X_train_t.shape[1]
model = NewsMLP(input_dim=input_dim, num_classes=num_classes).to(device)

# =========================
# Compile the Model (PyTorch-style setup)
# =========================
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

# =========================
# Train the Model
# =========================
def train(num_epochs=10):
    assert optimizer is not None and criterion is not None, "Set optimizer/criterion before training."
    for epoch in range(num_epochs):
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            # Metrics
            preds = logits.argmax(dim=1)
            running_loss += loss.item() * xb.size(0)
            running_correct += (preds == yb).sum().item()
            total += xb.size(0)

        epoch_loss = running_loss / max(1, total)
        epoch_acc  = running_correct / max(1, total)
        print(f"Epoch {epoch+1}/{num_epochs} | loss={epoch_loss:.4f} acc={epoch_acc:.4f}")

        if scheduler is not None:
            scheduler.step(epoch_loss)  # Reduce LR on plateau

# =========================
# Evaluate the Model
# =========================
def evaluate():
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    print("Test Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=data.target_names))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    # ===== Run training and evaluation =====
    train(num_epochs=50)
    evaluate()
