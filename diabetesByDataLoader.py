import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class DiabetesDataset(Dataset):
    def __init__(self, filename):
        # 读取数据
        xy = np.loadtxt(filename, delimiter=",", dtype=np.float32)
        # shape返回一个元组，为数据的(行,列)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        # 注意这里没有第二个:, 且应该加[], 这样取出来的数据是矩阵
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len


myDataset = DiabetesDataset("diabetes.csv.gz")
loader = DataLoader(dataset=myDataset, batch_size=8, shuffle=True, num_workers=4)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 降维
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 3)
        self.linear3 = torch.nn.Linear(3, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    for i, (inputs, labels) in enumerate(myDataset, 0):
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print("第%d轮, 第%d块, 损失值为%d" % (epoch, i, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



























