import torch
import torch.nn as nn
import torch.nn.functional as F

from OverFitter import OverFitter
from BiasedAdam import BiasedAdam
from Trainer import Trainer


class SparNetLSTM(nn.Module):
    def __init__(self, dims=128, dropout=0.0):
        # img width range: [110, 856]
        # pixels/char range: [7.070422535211268, 131.2]
        # num chars range: [6, 80]
        super(SparNetLSTM, self).__init__()
        self.dropout = dropout
        self.dims = dims

        # N, W, H=27+4
        # unsqueeze
        # N, 1, W, H
        '''self.conv = nn.Conv2d(1, round(dims / 32), kernel_size=(5, 5),
                              padding=(2, 2))  # in_channels, out_channels, kernel_size'''
        self.conv = nn.Conv2d(1, round(dims / 16), kernel_size=(5, 5),
                              padding=(2, 2))  # in_channels, out_channels, kernel_size (standard)
        '''self.conv = nn.Conv2d(1, round(dims / 8), kernel_size=(5, 5),
                              padding=(2, 2))  # in_channels, out_channels, kernel_size'''

        '''self.conv_1 = nn.Conv2d(1, round(dims / 4), kernel_size=(5, 5), padding=(2, 0))
        self.conv_2 = nn.Conv2d(round(dims / 4), round(dims / 4), kernel_size=(5, 5), padding=(2, 0))
        # in_channels, out_channels, kernel_size'''

        # N, ~8, W, H
        # 2x2 pooling
        # N, ~8, W/2, H/2
        # -> N, dims

        self.dense_i1 = nn.Linear(2 * dims, dims)
        self.dense_f1 = nn.Linear(2 * dims, dims)
        with torch.no_grad():
            self.dense_f1.bias.add_(1.0)
        self.dense_o1 = nn.Linear(2 * dims, dims)
        self.dense_g1 = nn.Linear(2 * dims, dims)
        self.c1 = torch.rand((1, dims), requires_grad=True)

        self.dense_i2 = nn.Linear(2 * dims, dims)
        self.dense_f2 = nn.Linear(2 * dims, dims)
        with torch.no_grad():
            self.dense_f2.bias.add_(1.0)
        self.dense_o2 = nn.Linear(2 * dims, dims)
        self.dense_g2 = nn.Linear(2 * dims, dims)
        self.c2 = torch.rand((1, dims), requires_grad=True)

        # N, 2*dims, W/2
        self.batch_norm = nn.BatchNorm1d(2 * dims, affine=False)
        # W/2, N, 2*dims
        self.dense_y = nn.Linear(2 * dims, 95, bias=False)
        # W/2, N, 95

        if 0.0 < self.dropout < 1.0:
            self.norm = nn.Dropout(self.dropout)
        else:
            self.norm = lambda x: x

    def forward(self, x):
        t = torch.unsqueeze(self.norm(x.float()), dim=1)

        t = self.conv(t)
        # t = torch.cat((t, -t), dim=1)
        t = self.norm(F.max_pool2d(t, kernel_size=2, padding=1))
        # t = self.norm(F.max_pool2d(t, kernel_size=2, padding=(1, 0)))

        '''t = self.norm(F.max_pool2d(F.relu(self.conv_1(t)), kernel_size=2, padding=(1, 0)))  # 13
        t = self.norm(F.max_pool2d(self.conv_2(t), kernel_size=(1, 2)))  # 4'''

        c1 = self.c1.expand((t.size()[0], self.c1.size()[1]))
        h1 = torch.tanh(c1)
        y1 = []
        for col in range(t.size()[2]):
            tt = torch.cat((h1, t[:, :, col, :].reshape(t.size()[0], -1)), dim=1)
            i = torch.sigmoid(self.dense_i1(tt))
            f = torch.sigmoid(self.dense_f1(tt))
            o = torch.sigmoid(self.dense_o1(tt))
            g = torch.tanh(self.dense_g1(tt))
            c1 = i * g + f * c1
            h1 = self.norm(o * torch.tanh(c1))
            y1.append(h1)

        c2 = self.c2.expand((t.size()[0], self.c2.size()[1]))
        h2 = torch.tanh(c2)
        y2 = []
        for col in range(t.size()[2] - 1, -1, -1):
            tt = torch.cat((h2, t[:, :, col, :].reshape(t.size()[0], -1)), dim=1)
            i = torch.sigmoid(self.dense_i2(tt))
            f = torch.sigmoid(self.dense_f2(tt))
            o = torch.sigmoid(self.dense_o2(tt))
            g = torch.tanh(self.dense_g2(tt))
            c2 = i * g + f * c2
            h2 = self.norm(o * torch.tanh(c2))
            y2.insert(0, h2)

        y = torch.cat((torch.stack(y1, dim=2), torch.stack(y2, dim=2)), dim=1)
        y = self.batch_norm(y).permute(0, 2, 1)
        return self.dense_y(y)


if __name__ == '__main__':
    filename = 'model_13.pth'
    print('Hello World, I am SparNet')
    '''trainer = OverFitter( 
        optimizer=lambda p: BiasedAdam(p, lr=0.001, log_estimates=True),
        network=SparNetLSTM(),
        num_datapoints=16,
    )'''
    trainer = Trainer(
        optimizer=lambda p: BiasedAdam(p, lr=0.01, log_estimates=True),
        network=SparNetLSTM(dims=256),
    )
    trainer.train(n_epochs=30, synflow=True)
    torch.save(trainer.network.state_dict(), filename)

    '''network = SparNetLSTM()
    network.load_state_dict(torch.load('model_10.pth'))
    trainer = Trainer(
        optimizer=lambda p: BiasedAdam(p, lr=0.01, log_estimates=True),
        network=network
    )
    trainer.train(n_epochs=24, synflow=True)
    torch.save(trainer.network.state_dict(), 'model_10_1.pth')'''