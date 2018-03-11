import torch
import random
from torch.autograd import Variable

class DynamicNet(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DynamicNet, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        h_relu = self.layer1(input).clamp(min=0)
        for _ in range(random.randint(1, 4)):
            h_relu = self.layer2(h_relu).clamp(min=0)
        y_pred = self.layer3(h_relu)
        return y_pred


batch_size = 64
input_dim = 500
hidden_dim = 100
output_dim = 7

x = Variable(torch.randn(batch_size, input_dim))
y = Variable(torch.randn(batch_size, output_dim), requires_grad=False)

model = DynamicNet(input_dim, hidden_dim, output_dim)
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

for _ in range(600):

    y_pred = model(x)

    loss = criterion(y_pred, y)
    print(_, loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
