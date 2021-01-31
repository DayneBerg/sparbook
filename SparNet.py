import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SparNet(nn.Module):
    def __init__(self, dims=64, dropout=0.5, max_len1=214, max_len2=80):
        # dim must be even for PE
        # max_len1 = 856 after pooling, max_len2 = 80
        super(SparNet, self).__init__()
        self.dims = dims
        self.max_len2 = max_len2
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=None)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=(0, 1))
        self.dsize1 = 2 ** math.ceil(0.5 * math.log(80 * dims, 2))  # 128
        self.dense1 = nn.Linear(80, self.dsize1)
        self.norm1 = LayerNorm(self.dsize1, dropout)
        self.dense2 = nn.Linear(self.dsize1, dims)
        self.norm2 = LayerNorm(dims, dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len1, dims)
        # max_len1, dim
        position = torch.arange(0, max_len1).unsqueeze(1)
        # max_len1, 1
        div_term = torch.exp(torch.arange(0, dims, 2) * -(math.log(10000.0) / dims))
        # ceil(dim/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # 1, max_len1, dims
        self.register_buffer('pe', pe)

        self.initQ = nn.Parameter(torch.randn(dims))
        self.denseQ1 = nn.Linear(dims, 2 * dims)
        self.normQ1 = LayerNorm(2 * dims, dropout)
        self.denseQ2 = nn.Linear(2 * dims, dims)
        self.normQ2 = LayerNorm(dims, dropout)

        self.dsize3 = 2 ** math.ceil(0.5 * math.log(dims * 95, 2))  # 128
        self.dense3 = nn.Linear(dims, self.dsize3)
        self.norm3 = LayerNorm(self.dsize3, dropout)
        self.dense4 = nn.Linear(self.dsize3, 95)

    def forward(self, x):  # N,1,31,W+4
        # ENCODING
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=2, ceil_mode=True))
        # N,8,27,PW-4 -> N,8,13,ceil(PW-1)/2
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=3, stride=2, ceil_mode=True))
        # N,16,11,PW -> N,16,5,ceil(PW-1)/2
        x = torch.stack([t.view(-1, 80) for t in torch.unbind(x, dim=3)], dim=1)
        # ((N, 16, 5),...) -> ((N, 80),...) -> N, W, 80
        x = self.norm1(F.relu(self.dense1(x)))
        # N, W, dsize1
        x = self.norm2(self.dense2(x) + Variable(self.pe[:, :x.size(1), :], requires_grad=False))
        # N, W, dims

        # DECODING
        y = torch.zeros(x.size(0), self.dims, self.max_len2)
        # N, max_len2, dims
        y[:, 0, :] = attention(self.initQ, x, x)
        for i in range(1, self.max_len2):
            y[:, i, :] = attention(
                query=self.normQ2(y[:, i - 1, :] + self.denseQ2(self.normQ1(F.relu(self.denseQ1(y[:, i - 1, :]))))),
                key=x,
                value=x
            )
        y = self.norm3(F.relu(self.dense3(y)))
        # N, max_len2, dsize3
        y = self.dense4(y)
        # N, max_len2, 95
        # for benchmarking, -torch.log(torch.nextafter(torch.zeros(1), torch.ones(1)))
        return F.log_softmax(y, dim=2)


class LayerNorm(nn.Module):
    def __init__(self, size, dropout=0.5, eps=1e-6, dim=-1):
        super(LayerNorm, self).__init__()
        self.weights = nn.Parameter(torch.ones(size))
        self.biases = nn.Parameter(torch.zeros(size))
        self.drop = nn.Dropout(dropout)
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        mean = x.mean(dim=self.dim, keepdim=True)
        std = x.std(dim=self.dim, keepdim=True)
        x = self.weights * (x - mean) / (std + self.eps) + self.biases
        return self.drop(x)


def attention(query, key, value):  # , mask=None):
    """Compute Scaled Dot Product Attention"""
    d_k = query.size(-1)  # dims
    relv = torch.matmul(query, key) / math.sqrt(d_k)
    # N, W
    '''if mask is not None:
        relv = relv.masked_fill(mask == 0, -1e9)'''
    attn = F.softmax(relv, dim=-1)
    return torch.matmul(attn, value.transpose(-2, -1)), attn  # N, dims

class Trainer:
    def __init__(self):
        # 761 lines of data, 45891 characters of data (~64 chars per line)
        batch_size_train = 2  # ~128 chars
        batch_size_test = 4  # ~256 chars
        torch.backends.cudnn.enabled = False  # forces random number generation to be deterministic
        random_seed = 144
        torch.manual_seed(random_seed)

        def train(epoch):
            network.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = network(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len_train_set,
                               100. * batch_idx / len(train_loader), loss.item()))
                    train_losses.append(loss.item())
                    train_counter.append(
                        (batch_idx * batch_size_train) + ((epoch - 1) * len_train_set))
                    torch.save(network.state_dict(), 'C:/Users/Dayne/Documents/MNIST/results/model.pth')
                    torch.save(optimizer.state_dict(), 'C:/Users/Dayne/Documents/MNIST/results/optimizer.pth')

        n_epochs = 8
        n_networks = 8
        learning_rate = 0.001
        betas = (0.9, 0.999)


if __name__ == '__main__':
    q = [0.1, 2.3]
    w = [4.5, 6.7]
    print(q + w)
