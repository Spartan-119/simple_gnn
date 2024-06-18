import torch

def train_model(model, data, train_mask, val_mask, optimizer, criterion, num_epochs = 200):
    model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                val_acc = torch.sum(out[val_mask].argmax(dim = 1) == data.y[val_mask]) / val_mask.sum()
                print(f'Epoch: {epoch}, Validation Accuracy: {val_acc:.4f}')
    
    return model