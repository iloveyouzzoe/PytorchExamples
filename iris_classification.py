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
        self.layer2 = torch.nn.Linear(hidden, output)

    def forward(self, input):
        out1 = self.layer1(input)
        relu1 = out1.clamp(min=0)
        y_pred = self.layer2(relu1)
        return y_pred


class IrisTrainData(Dataset):

    def __init__(self, path):
        df = pd.read_csv(path, delimiter=',')
        df.replace(to_replace='Iris-setosa', value=1, inplace=True)
        df.replace(to_replace='Iris-versicolor', value=2, inplace=True)
        df.replace(to_replace='Iris-virginica', value=3, inplace=True)
        self.X_train = df.as_matrix()[:-10, :-1]
        self.y_train = df.as_matrix()[:-10, -1:]

    def __getitem__(self, item):
        data = torch.from_numpy(self.X_train[item])
        one_hot = [0] * 3
        one_hot[int(self.y_train[item, 0]-1)] = 1
        label = torch.from_numpy(np.array(one_hot))
        return data, label


if __name__ == "__main__":
    path = 'C:\\Users\\Swapnil.Walke\\PycharmProjects\\PytorchExamples\\iris_data.csv'
    train_dataset = IrisTrainData(path)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=5, num_workers=1)

    inp = 4
    hidden  = 50
    out = 3

    model = IrisModel(inp, hidden, out)
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for _ in range(10):
        for index, (data, label) in enumerate(train_dataset):
            data = Variable(data.type(torch.FloatTensor))
            label = Variable(label.type(torch.FloatTensor).view(1,3))
            y_pred = model(data)
            loss = criterion(y_pred.view(1,3), label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()