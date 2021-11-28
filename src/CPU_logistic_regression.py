import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import collections


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out


def _load_train_dataset():
    train_dataset = dsets.MNIST(
        root='./data', 
        train=True, 
        transform=transforms.ToTensor(),
        download=True
    )
    return train_dataset


def _load_test_dataset():
    test_dataset = dsets.MNIST(
        root='./data', 
        train=False, 
        transform=transforms.ToTensor()
    )
    return test_dataset


def _create_iterable_obj(dataset):
    batch_size = 100
    n_iters = 3000
    num_epochs = n_iters / (len(dataset) / batch_size)
    num_epochs = int(num_epochs)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    assert isinstance(train_loader, collections.Iterable)
    return train_loader


def _init_model_class():
    input_dim = 28*28
    output_dim = 10
    return LogisticRegressionModel(input_dim, output_dim)


def _train_model(num_epochs, train_loader, test_loader, model, criterion, optimizer):
    iter = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Load images as Variable
            images = images.view(-1, 28*28).requires_grad_()
            labels = labels
            
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            
            # Forward pass to get output/logits
            outputs = model(images)
            
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            
            # Getting gradients w.r.t. parameters
            loss.backward()
            
            # Updating parameters
            optimizer.step()
            
            iter += 1
            
            if iter % 500 == 0:
                # Calculate Accuracy         
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    # Load images to a Torch Variable
                    images = images.view(-1, 28*28).requires_grad_()
                    
                    # Forward pass only to get logits/output
                    outputs = model(images)
                    
                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)
                    
                    # Total number of labels
                    total += labels.size(0)
                    
                    # Total correct predictions
                    correct += (predicted == labels).sum()
                
                accuracy = 100 * correct / total
                
                # Print Loss
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))


if __name__ == "__main__":
    """
    Step 1: Load Dataset
    Step 2: Make Dataset Iterable
    Step 3: Create Model Class
    Step 4: Instantiate Model Class
    Step 5: Instantiate Loss Class
    Step 6: Instantiate Optimizer Class
    Step 7: Train Model
    """
    train_dataset = _load_train_dataset()
    test_dataset = _load_test_dataset()
    train_iter = _create_iterable_obj(train_dataset)
    test_iter = _create_iterable_obj(test_dataset)
    model = _init_model_class()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  
    _train_model(train_iter, test_iter, model, criterion, optimizer)