import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  
    
    def forward(self, x):
        out = self.linear(x)
        return out


def _plot_linear_regression():
    np.random.seed(3)
    n = 50
    x = np.random.randn(n)
    y = x * np.random.randn(n)

    colors = np.random.rand(n)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

    plt.scatter(x, y, c=colors, alpha=0.5)
    plt.show()


def _build_dataset():
    x_values = [i for i in range(11)]
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [2*i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32).reshape(-1, 1)
    return x_train, y_train


def _train_model(x_train, y_train):

    # Instantiate Model Class
    input_dim = 1
    output_dim = 1
    model = LinearRegressionModel(input_dim, output_dim)
    # Instantiate Loss Class
    criterion = nn.MSELoss()
    # Instantiate Optimizer Class
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # Train Model
    epochs = 100
    for epoch in range(epochs):
        epoch += 1
        # Convert numpy array to torch Variable
        inputs = torch.from_numpy(x_train).requires_grad_()
        labels = torch.from_numpy(y_train)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad() 
        
        # Forward to get output
        outputs = model(inputs)
        
        # Calculate Loss
        loss = criterion(outputs, labels)
        
        # Getting gradients w.r.t. parameters
        loss.backward()
        
        # Updating parameters
        optimizer.step()
        
        print('epoch {}, loss {}'.format(epoch, loss.item()))
    
    return model


def _plot_graph(mode, x_train, y_train):
    # Clear figure
    plt.clf()

    # Get predictions
    predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()

    # Plot true data
    plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)

    # Plot predictions
    plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)

    # Legend and plot
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    """
    Aim of Linear Regression:
    - Minimize the distance between the points and the line y = alpha x + beta
    - Adjusting
        - Coefficient: alpha
        - Bias/intercept: beta
    """
    _plot_linear_regression()
    x_train, y_train = _build_dataset()
    model = _train_model(x_train, y_train)
    # Compare results
    predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
    print(predicted)
    print(y_train)
    _plot_graph(model, x_train, y_train)
    torch.save(model.state_dict(), 'linear_regression_model.pkl')
