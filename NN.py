import main as m
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import numpy as np
import torch.utils.data as utils
from sklearn.model_selection import train_test_split

# The parts that you should complete are designated as TODO
class Module(nn.Module):
    def __init__(self,n_dims):
        super(Module, self).__init__()
        # TODO: define the layers of the network
        self.lin1 = nn.Linear(n_dims,2048)
        self.lin2 = nn.Linear(2048,64)
        self.lin3 = nn.Linear(64,8)
        self.out = nn.Linear(8,1)

    def forward(self, x):
        # TODO: define the forward pass of the network using the layers you defined in constructor
        x=F.relu(self.lin1(x))
        x=F.relu(self.lin2(x))
        x=F.sigmoid(self.lin3(x))
        x=self.out(x)
        return x

def main():
    torch.manual_seed(1)
    np.random.seed(1)
    # Training settings
    use_cuda = True # Switch to False if you only want to use your CPU
    learning_rate = 0.01
    NumEpochs = 100

    device = torch.device("cuda" if use_cuda else "cpu")
    
    data = m.get_stock_data('SPY')
    data = m.get_indicator_data(data)
    data = data.iloc[16:]
    data = data.dropna()
    X,y = m.get_x_y(data)
    today_data = data.iloc[-1:]
    del (today_data['open'])
    del (today_data['close'])
    _,n_dims = X.shape

    train_X, test_X, train_Y, test_Y = train_test_split(X,y,test_size=0.33)

    train_X = train_X.astype('float32')
    train_Y = train_Y.astype('int64')
    test_X = test_X.astype('float32')
    test_Y = test_Y.astype('int64')

    train_X = torch.from_numpy(train_X).type(torch.Tensor).to(device)
    train_Y = torch.from_numpy(train_Y).type(torch.Tensor).to(device)
    test_X = torch.from_numpy(test_X).type(torch.Tensor).to(device)
    test_Y = torch.from_numpy(test_Y).type(torch.Tensor).to(device)
    
    model = Module(n_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse = torch.nn.MSELoss(reduction='mean')

    train_mse = []
    test_mse =[]

    for epoch in range(NumEpochs):
        y_train_pred = model(train_X)
        y_test_pred = model(test_X)
        loss = torch.sqrt(mse(y_train_pred, train_Y))
        print("Epoch ", epoch, "MSE: ", loss.item())
        train_mse.append(loss.item())
        test_loss = torch.sqrt(mse(y_test_pred,test_Y))
        test_mse.append(test_loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "mnist_cnn.pt")
    
    #TODO: Plot train and test accuracy vs epoch
    plt.plot(range(NumEpochs), train_mse, label= 'Train Acc')
    plt.plot(range(NumEpochs), test_mse, label = 'Test Acc')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()