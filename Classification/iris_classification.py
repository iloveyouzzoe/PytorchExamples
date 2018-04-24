import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from torch.autograd import Variable


class IrisModel(torch.nn.Module):

    def __init__(self, input, hidden, output):
        super(IrisModel, self).__init__()
        self.layer1 = torch.nn.Linear(input, hidden)
        self.layer2 = torch.nn.Linear(hidden, hidden)
        self.layer3 = torch.nn.Linear(hidden, output)

    def forward(self, input):
        sigmoid = torch.nn.Tanh()
        relu = torch.nn.ReLU()
        out1 = self.layer1(input)
        out2 = self.layer2(relu(out1))
        y_pred = self.layer3(relu(out2))
        return y_pred


class IrisTestData(Dataset):

    def __init__(self, path):
        df = pd.read_csv(path, delimiter=',')
        df.replace(to_replace='Iris-setosa', value=0, inplace=True)
        df.replace(to_replace='Iris-versicolor', value=1, inplace=True)
        df.replace(to_replace='Iris-virginica', value=2, inplace=True)
        df = df.sample(frac=1)
        self.X_train = df.as_matrix()[-10:, :-1]
        self.y_train = df.as_matrix()[-10:, -1:]

    def __len__(self):
        return len(self.X_train[:, 0:])

    def __getitem__(self, item):
        data = torch.from_numpy(self.X_train[item])
        label = torch.from_numpy(self.y_train[item])
        return data, label


class IrisTrainData(Dataset):

    def __init__(self, path):
        df = pd.read_csv(path, delimiter=',')
        df.replace(to_replace='Iris-setosa', value=0, inplace=True)
        df.replace(to_replace='Iris-versicolor', value=1, inplace=True)
        df.replace(to_replace='Iris-virginica', value=2, inplace=True)
        df = df.sample(frac=1)
        self.X_train = df.as_matrix()[:-10, :-1]
        self.y_train = df.as_matrix()[:-10, -1:]

    def __len__(self):
        return len(self.X_train[:, 0:])

    def __getitem__(self, item):
        data = torch.from_numpy(self.X_train[item])
        label = torch.from_numpy(self.y_train[item])
        return data, label


if __name__ == "__main__":
    torch.manual_seed(100)
    np.random.seed(100)
    path = 'C:\\Users\\Swapnil.Walke\\PycharmProjects\\PytorchExamples\\iris_data.csv'
    train_dataset = IrisTrainData(path)
    test_dataset = IrisTestData(path)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=1)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1, num_workers=1)
    inp = 4
    hidden = 50
    out = 3

    model = IrisModel(inp, hidden, out)
    # cross entropy loss needs 1D target vector
    # squeeze the target vector before passing to criterion
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for _ in range(15):
        for index, (data, label) in enumerate(train_loader):
            data = Variable(data.type(torch.FloatTensor))
            label = Variable(label.type(torch.LongTensor))
            label = label.squeeze(1)
            # print(label)
            y_pred = model(data)
            loss = criterion(y_pred, label)
            print(f'Train Loss {loss.data[0]} : {_}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for index, (data, label) in enumerate(test_loader):
                data = Variable(data.type(torch.FloatTensor))
                label = Variable(label.type(torch.LongTensor))
                y_pred = model(data)
                label = label.squeeze(1)
                test_loss = criterion(y_pred, label)
                print(f'Test Loss {_} : {test_loss.data[0]}')
                break
            break
    cnt = 0.0
    tot = 0
    for index, (data, label) in enumerate(test_loader):
        tot += 1
        data = Variable(data.type(torch.FloatTensor))
        label = Variable(label.type(torch.FloatTensor))
        y_pred = model(data)
        a, ind1 = torch.max(y_pred, 1)
        b, ind2 = torch.max(label, 1)
        if ind2.data[0] == ind1.data[0]:
            cnt += 1
    print(cnt)
    print((cnt/tot)*100)