import pandas as pd
import torch
from torch.nn import Module
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import numpy as np


class DiabetesTrain(Dataset):

    def __init__(self, path):
        super(Dataset, self).__init__()
        self.path = path
        self.dataset = pd.read_csv(self.path, delimiter=",")
        self.dataset = self.dataset.sample(frac=1)
        self.dataset.replace(to_replace='M', value=0, inplace=True)
        self.dataset.replace(to_replace='B', value=1, inplace=True)
        self.data = self.dataset.iloc[:-100, :-1].as_matrix()
        self.label = self.dataset.iloc[:-100, -1].as_matrix()

    def __getitem__(self, item):
        return self.data[item, :], self.label[item]

    def __len__(self):
        return len(self.data[:, :])


class DiabetesTest(Dataset):

    def __init__(self, path):
        super(Dataset, self).__init__()
        self.path = path
        self.dataset = pd.read_csv(self.path, delimiter=",")
        self.dataset.replace(to_replace='M', value=0, inplace=True)
        self.dataset.replace(to_replace='B', value=1, inplace=True)
        self.dataset = self.dataset.sample(frac=1)
        self.data = self.dataset.iloc[-100:, :-1].as_matrix()
        self.label = self.dataset.iloc[-100:, -1].as_matrix()

    def __getitem__(self, item):
        return self.data[item, :], self.label[item]

    def __len__(self):
        return self.dataset.shape[0]


class Model(Module):

    def __init__(self, input, hidden, output):
        Module.__init__(self)
        self.layer1 = torch.nn.Linear(input, hidden)
        self.layer2 = torch.nn.Linear(hidden, output)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        out1 = self.relu(self.layer1(input))
        out2 = self.layer2(out1)
        return out2


if __name__ == "__main__":

    # seed values
    np.random.seed(100)
    torch.manual_seed(777)

    # Model params
    input = 13
    hidden = 27
    output = 2
    batch_size = 32

    # Load train and test data set
    diabetes_train = DiabetesTrain(path="C:\\Users\\Swapnil.Walke\\PycharmProjects\\PytorchExamples\\diabetes.csv")
    diabetes_test = DiabetesTest(path="C:\\Users\\Swapnil.Walke\\PycharmProjects\\PytorchExamples\\diabetes.csv")
    diabetes_train = DataLoader(diabetes_train, batch_size=batch_size, shuffle=True, num_workers=1)
    diabetes_test = DataLoader(diabetes_test, batch_size=batch_size, shuffle=True, num_workers=1)

    # initialize model, criterion and optimizer
    model = Model(input, hidden, output)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(15):
        loss = 0.0
        for index, data in enumerate(diabetes_train):
            data, label = data
            x = Variable(data.float(), requires_grad=True)
            y = Variable(label)
            y_pred = model(x)
            optimizer.zero_grad()
            loss += criterion(y_pred, y)
            loss.backward(retain_graph=True)
            optimizer.step()
        loss = (loss/batch_size)
        print(f' Iteration {epoch} : {loss.data[0]}')