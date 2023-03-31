import torch


# 反向传播
# x_data = [1, 2, 3]
# y_data = [3, 6, 9]
#
# w = torch.tensor([1.0], requires_grad=True)
#
# def forward(x):
#     return x * w
#
# def loss(x, y):
#     y_pred = forward(x)
#     return (y - y_pred) ** 2
#
# for epoch in range(100):
#     for x, y in zip(x_data, y_data):
#         l = loss(x, y)
#         l.backward()
#         print("grad:\t", x, y, w.grad.item())
#         w.data -= 0.1 * w.grad.data
#
#         w.grad.data.zero_()
#
#     print("process:\t", epoch, l.item())

#
# class person():
#     def __call__(self, *args, **kwargs):
#         pass
#
# class test(person):
#     def testfunc1(self, x):
#         print(x)
#
#     # def testfunc2(self, x):
#     #     print(x)
#
#
# test1 = test()
# print(test1(5))

import torch

x_data = torch.tensor([[1.], [2.], [3.]])
y_data = torch.tensor([[2.], [4.], [6.]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.linear.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w= ', model.linear.weight.item())
print('b= ', model.linear.bias.item())

x_test = torch.tensor([[4.]])
y_test_pred = model(x_test)
print(y_test_pred.item())


























