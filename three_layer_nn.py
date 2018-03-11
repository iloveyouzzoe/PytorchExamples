import torch
from torch.autograd import Variable


class ThreeLayerNet(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(ThreeLayerNet, self).__init__()
        self.layer_1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.layer_2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.layer_3 = torch.nn.Linear(hidden_dim2, output_dim)

    def forward(self, input):
        out1 = self.layer_1(input)
        relu1 = out1.clamp(min=0)
        out2 = self.layer_2(relu1)
        relu2 = out2.clamp(min=0)
        y_pred = self.layer_3(relu2)
        return y_pred


batch_size = 64
input_dim = 500
hidden_dim1 = 100
hidden_dim2 = 60
output_dim = 5

x = Variable(torch.randn(batch_size, input_dim))
y = Variable(torch.randn(batch_size, output_dim), requires_grad=False)

model = ThreeLayerNet(input_dim, hidden_dim1, hidden_dim2, output_dim)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model. parameters(), lr=1e-4)

for _ in range(500):

    y_pred = model(x)

    loss = criterion(y_pred, y)

    print(_, loss.data[0])

    # update grads to zero
    optimizer.zero_grad()
    # calculate grads
    loss.backward()
    # update weights
    optimizer.step()