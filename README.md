![Python application test with github actions](https://github.com/YXTanDuke/IDS706_Group_Project/actions/workflows/main.yml/badge.svg)

## About The Project

This is the Group Project for IDS 706 Fall 2021, group members include Chang Xu and Yongxin Tan. 

As a subset of Artificial Intelligence (AI), machine learning (ML) is the area of computational science that focuses on analyzing and interpreting patterns and structures in data to enable learning, reasoning, and decision making outside of human interaction. After writing a machine learning model, a user can feed the algorithm countless data that none human is able to process, so that the algorithm will makr data-driven recommendataions and decisions. 

Although machine learning algorithm usually requires tremendous computing power, cloud computing provides nearly unlimite computing resources so that nearly everyone can train a ML model for free. The main goal of this project is to show you both the code and a step-by-step tutorial to train a model on Google Cloud Platform (GCP). You can find all necessary code under the `/src` folder, and this document will focus more on the tutorial part.

In this document, we will provide the following contents:
* Basic matrix operations
* Basic data structures for PyTorch
* How to write a linear regression model
* How to train a XXX model using Google Colab
* Resource links for each topic

## Getting Started

### Local Installation

If you want to play with the code locally, Python is the only thing you need. 

1. Start by installing Homebrew:

    ```
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```
2. Install Pyenv using Homebrew to manage your Python version:

    ```
    brew install pyenv
    ```
3. Install Python using Pyenv, we recommend using Python 3.9:

    ```
    pyenv install 3.9.2 
    ```
4. To test that you have successfully installed Python, run the following code and you should get your version number:
    ```
    python3 --version
    ```

For troubleshooting and more info, please refer to the following link: (https://docs.python.org/3/using/mac.html)

### Colab Getting Started

To get started with Google Colab, please refer to the following video

https://www.youtube.com/watch?v=inN8seMm7UI

## Matrix Basics

### TODO @Yongxin Tan

## PyTorch Basics

### TODO @Yongxin Tan

## Write code for a Linear Regression Model

Linear regression is a basic and commonly used type of predictive analysis. These regression estimates are used to explain the relationship between one dependent variable and one or more independent variables. The aim of Linear Regression is to minimize the distance between the points and the line y=αx+β. 

* α is the coefficients and β is the bias/intercept.

To write code for a Linear Regression Model, you can perform the following steps:

1. Build Dataset
    ```
    def _build_dataset():
        x_values = [i for i in range(11)]
        x_train = np.array(x_values, dtype=np.float32)
        x_train = x_train.reshape(-1, 1)

        y_values = [2*i + 1 for i in x_values]
        y_train = np.array(y_values, dtype=np.float32).reshape(-1, 1)
        return x_train, y_train
    ```
2. Create Model Class
    ```
    class LinearRegressionModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)  
        
        def forward(self, x):
            out = self.linear(x)
            return out
    ```
3. Instantiate Model Class
    ```
    model = LinearRegressionModel(input_dim, output_dim)
    ```
4. Instantiate Loss Class
    ```
    criterion = nn.MSELoss()
    ```
5. Instantiate Optimizer Class
    ```
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    ```
6. Train the Model
    ```
    for epoch in range(epochs):
        epoch += 1
        inputs = torch.from_numpy(x_train).requires_grad_()
        labels = torch.from_numpy(y_train)
        optimizer.zero_grad() 
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))
    ```

More detailed code can be found in `/src/CPU_linear_regression.py`

## Train a model on Google Colab

### TODO @Chang Xu
