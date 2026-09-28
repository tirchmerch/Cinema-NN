import pandas as pd
from CNN_Model import CNN
from sklearn.model_selection import train_test_split

# Load the CSV file into a DataFrame

df = pd.read_csv('data.csv')
df = df[["quote", "score", "review_type"]]
df = df.rename(columns={"quote": "text", "score": "label", "review_type": "type"})


# Split the data into training, validation, and test sets using train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Parse through a small amount of data (100 reviews) to test if the CNN works, REMOVE this when it's working
train.sample(100, random_state=42)[["text", "label", "type"]]


# Build train validation test data loaders using pytorch

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Build model architecture (IN A DIFFERENT FILE)
model = CNN(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

# Do weight initialization
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

# Training evaluation and hyperparameter tuning
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples

# Full training function
def fit_model(
    model,
    train_loader,
    val_loader,
    lr=0.1,
    weight_decay=5e-4,
    epochs=20,
    device=DEVICE,
    verbose=True,
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9,
        nesterov=True, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [], "lr": []
    }
    best_val_acc = -float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(current_lr))

        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train acc {100*train_acc:.2f}% | "
                f"val acc {100*val_acc:.2f}% | "
                f"lr {current_lr:.5f}"
            )

        scheduler.step()

    model.load_state_dict(best_state)
    return history


