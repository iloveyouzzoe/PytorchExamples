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
        return len(self.data[:, :])


class Model(Module):

    def __init__(self, input, hidden, output):
        Module.__init__(self)
        self.layer1 = torch.nn.Linear(input, hidden)
        self.layer1_5 = torch.nn.Linear(hidden, 23)
        self.layer2 = torch.nn.Linear(23, output)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        out1 = self.relu(self.layer1(input))
        out2 = self.relu(self.layer1_5(out1))
        return self.sigmoid(self.layer2(out2))


if __name__ == "__main__":

    # seed values
    np.random.seed(100)
    torch.manual_seed(888)

    # Model params
    input = 13
    hidden = 27
    output = 1
    batch_size = 32

    # Load train and test data set
    diabetes_train = DiabetesTrain(path="C:\\Users\\Swapnil.Walke\\PycharmProjects\\PytorchExamples\\diabetes_norm.csv")
    diabetes_test = DiabetesTest(path="C:\\Users\\Swapnil.Walke\\PycharmProjects\\PytorchExamples\\diabetes_norm.csv")
    diabetes_train = DataLoader(diabetes_train, batch_size=batch_size, shuffle=True, num_workers=1)
    diabetes_test = DataLoader(diabetes_test, batch_size=1, shuffle=True, num_workers=1)

    # initialize model, criterion and optimizer
    model = Model(input, hidden, output)
    criterion = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    for epoch in range(20):
        loss = 0.0
        for index, data in enumerate(diabetes_train):
            data, label = data
            x = Variable(data.float(), requires_grad=True)
            y = Variable(label.float())
            y_pred = model(x)
            optimizer.zero_grad()
            loss += criterion(y_pred, y)
            loss.backward(retain_graph=True)
            optimizer.step()
        train_loss = (loss/len(diabetes_train))
        loss = 0
        for index, data in enumerate(diabetes_test):
            data, label = data
            x = Variable(data.float())
            y = Variable(label.float())
            y_pred = model(x)
            loss += criterion(y_pred, y)
        test_loss = (loss/len(diabetes_test))
        print(f'{epoch} : train [{train_loss.data[0]}] Test [{test_loss.data[0]}]')

    # Calculate the accuracy
    cnt = 0.0
    tot = 0
    for index,data in enumerate(diabetes_test):
        data, label = data
        x = Variable(data.float())
        y = Variable(label.float())
        y_pred = model(x)
        pred = y_pred.data.numpy()[0][0]
        var = y.data.numpy()[0]
        tot += 1
        if pred > 0.5:
            pred = 1.0
        else:
            pred = 0.0
        if pred == var:
            cnt += 1
    print(f'Accuracy : {cnt/tot}')