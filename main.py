import torch

from gnn.model import GNN
from gnn.data import load_data
from gnn.train import train_model
from gnn.evaluate import evaluate_model

def main():
    # Load data
    data, train_mask, val_mask, test_mask = load_data()

    # Get the number of classes
    num_classes = data.y.max().item() + 1  # Assuming classes are indexed from 0

    # Define model, optimizer, and loss function
    model = GNN(data.num_node_features, 16, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    trained_model = train_model(model, data, train_mask, val_mask, optimizer, criterion)

    # Evaluate the model
    evaluate_model(trained_model, data, test_mask)

if __name__ == "__main__":
    main()
