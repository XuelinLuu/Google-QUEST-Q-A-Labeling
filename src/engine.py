import torch
import torch.nn as nn

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)

def train_fn(model, device, data_loader, optimizer, scheduler=None):
    model.train()
    for i, data in enumerate(data_loader):

        ids = data['ids'].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        targets = data["targets"].to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids, attention_mask, token_type_ids)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if i % 50 == 0:
            print(f"train===>batch ids={i}, loss={loss}")

def eval_fn(model, device, data_loader):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)

            outputs = model(ids, attention_mask, token_type_ids)
            loss = loss_fn(outputs, targets)

            if i % 50 == 0:
                print(f"eval===>batch ids={i}, loss={loss}")