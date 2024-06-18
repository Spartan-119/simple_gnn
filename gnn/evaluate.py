import torch

def evaluate_model(model, data, test_mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_acc = torch.sum(out[test_mask].argmax(dim=1) == data.y[test_mask]) / test_mask.sum()
        print(f'Test Accuracy: {test_acc:.4f}')
