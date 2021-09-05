import math
from random import randint

import torch
import torch.nn.functional as F
from torch import tensor, square, norm, dot, flatten, sqrt

from SparbookDataset import SparbookDataset


def pad_seq(batch):  # requires W,H
    inputs = [x[0] for x in batch]
    # consider not using input_lengths to promote resilience to long empty tails
    half_input_lengths = torch.LongTensor([math.ceil(len(x) / 2) for x in inputs])
    # half_input_lengths = torch.LongTensor([math.ceil(max(len(x) for x in inputs)/2)] * len(batch))
    input_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)

    targets = [x[1] for x in batch]
    target_lengths = torch.LongTensor([len(x) for x in targets])
    target_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

    return input_padded, target_padded, half_input_lengths, target_lengths


def binomial_loss(output, output_lengths, target_lengths):
    """
    Assumes the separator character is index zero
    """
    x = torch.exp(output)
    t = x[:, :-1, 1:]
    t = torch.cat((torch.zeros((t.shape[0], 1, t.shape[2])), t), dim=1)
    t = torch.cat((torch.ones((t.shape[0], t.shape[1], 1)), t), dim=2)
    t = 1 - torch.sum(torch.mul(t, x), dim=2)
    means = torch.stack([torch.sum(t[i, :output_lengths[i]]) for i in range(t.shape[0])])
    t = torch.mul(t, 1 - t)
    variances = torch.stack([torch.sum(t[i, :output_lengths[i]]) for i in range(t.shape[0])])
    losses = variances + torch.square(target_lengths - means)
    return torch.mean(losses)


def calc_metric(params):
    p_sqr = tensor(0.0)
    grad_sqr = tensor(0.0)
    dot_prod = tensor(0.0)
    n_params = 0
    for p in params:
        if p.grad is not None:
            data = p.data
            grad = p.grad.data
            p_sqr.add_(square(norm(data)))
            grad_sqr.add_(square(norm(grad)))
            dot_prod.add_(dot(flatten(data), flatten(grad)))
            n_params += p.numel()
    sim = dot_prod / sqrt(p_sqr * grad_sqr)
    return dot_prod, sim, sim * n_params


def calc_acc(output, target_padded, target_lengths):
    output = torch.argmax(output, dim=2)
    num_correct = 0
    for i in range(output.shape[0]):  # batch
        pred = []
        for j in range(output.shape[1]):  # column
            if len(pred) == 0 or pred[-1] != output[i][j]:
                pred.append(output[i][j])
        pred = [e for e in pred if e > 0]
        for j in range(min(len(pred), target_lengths[i])):
            if pred[j] == target_padded[i][j]:
                num_correct += 1
    return num_correct


class Trainer:
    def __init__(self,
                 optimizer,
                 network,
                 ):
        torch.backends.cudnn.enabled = False  # forces random number generation to be deterministic
        # currently I do not set a specifc seed
        # 761 lines of data, max 856 pix per line, max 80 chars per line
        # target accuracy is either 99.18% or 99.8856% for 0.5 errors per line or per page

        '''self.dataset_size_test = 385  # ~ (8/3)*(761)^(3/4)
        self.batch_size_test = 385  # Should evenly divide dataset_size_test.
        self.dataset_size_train = 376  # = 761-385
        self.batch_size_train = 8  # ~ cube root of 376, open to experimentation
        # 47 batches per train epochs'''

        self.dataset_size_test = 32  # ~> (761)^(1/2)
        self.batch_size_test = 32  # Should evenly divide dataset_size_test.
        self.dataset_size_train = 729  # = 761-32
        self.batch_size_train = 16
        # self.batch_size_train = 9  # ~ cube root of 729 , open to experimentation
        # 81 batches per train epochs

        self.log_interval = 0.25
        self.cur_epoch = 0
        self.network = network
        self.optimizer = optimizer(self.network.parameters())
        dataset = SparbookDataset()
        # self.freq_table = dataset.freq_table
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [self.dataset_size_train, self.dataset_size_test]
        )
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size_train,
            shuffle=True,
            collate_fn=pad_seq,
            drop_last=True  # in case dataset does not evenly divide
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size_test,
            collate_fn=pad_seq,
        )

    def train(self, n_epochs, synflow=False):
        """
        TODO: train multiple networks in parallel
        TODO: calc train duration
        """
        if self.cur_epoch == 0:
            self.test_once(synflow=synflow)
        for epoch in range(n_epochs):
            self.cur_epoch += 1
            for batch_idx, (input_padded, target_padded, output_lengths, target_lengths) in enumerate(
                    self.train_loader):
                self.train_once(batch_idx, input_padded, target_padded, output_lengths, target_lengths)
            self.test_once(synflow=synflow)

    def train_once(self, batch_idx, input_padded, target_padded, output_lengths, target_lengths):
        num_batches = self.dataset_size_train // self.batch_size_train
        denominator = self.log_interval * num_batches
        self.network.train(mode=True)
        # input should be N,W,27+4
        '''print(input_padded.shape)
        assert (len(input_padded.shape) == 3), "input dimensions were incorrect (1)"
        assert (input_padded.shape[0] == self.batch_size_train), "input dimensions were incorrect (2)"
        assert (input_padded.shape[2] == 31), "input dimensions were incorrect (3)"'''
        self.optimizer.zero_grad()
        output = F.log_softmax(self.network(input_padded), dim=2)
        # bias = torch.cat((torch.tensor([1.0]), torch.tensor(self.freq_table)))
        # bias = torch.reshape(bias / torch.sum(bias), (1, 1, 95)).expand(output.size()[1], output.size()[0], 95)

        # N, W/2, 95
        loss = F.ctc_loss(output.permute(1, 0, 2), target_padded, output_lengths, target_lengths)
        # loss = loss + 0 * binomial_loss(output, output_lengths, target_lengths)
        loss.backward()
        self.optimizer.declare(loss)
        self.optimizer.step()
        num_correct = calc_acc(output, target_padded, target_lengths)
        num_chars = sum(target_lengths)
        if math.floor(batch_idx / denominator) > math.floor((batch_idx - 1) / denominator):
            print('Epoch {} ({}/{}) {:.0f}%\t\tLoss: {:.4f}\tAcc: ({}/{}) {:.0f}%'.format(
                self.cur_epoch,
                batch_idx,
                num_batches,
                100. * batch_idx / num_batches,
                loss.item(),
                num_correct,
                num_chars,
                100. * num_correct / num_chars
            ))
            print(self.optimizer.log)

    def test_once(self, synflow=False):
        num_test_batches = self.dataset_size_test // self.batch_size_test
        num_correct = 0
        num_chars = 0
        test_loss = 0
        self.network.train(mode=False)
        self.optimizer.zero_grad()
        for test_batch_idx, (input_padded, target_padded, output_lengths, target_lengths) \
                in enumerate(self.test_loader):
            if synflow:
                output = F.log_softmax(self.network(input_padded), dim=2)
                test_loss = test_loss + F.ctc_loss(
                    output.permute(1, 0, 2),
                    target_padded,
                    output_lengths,
                    target_lengths
                )
                # test_loss = test_loss + 0 * binomial_loss(output, output_lengths, target_lengths)
                test_loss.backward()
                syn = calc_metric(self.network.parameters())
                num_correct += calc_acc(output, target_padded, target_lengths)
                num_chars += sum(target_lengths)
            else:
                with torch.no_grad():
                    output = F.log_softmax(self.network(input_padded), dim=2)
                    test_loss = test_loss + F.ctc_loss(
                        output.permute(1, 0, 2),
                        target_padded,
                        output_lengths,
                        target_lengths
                    )
                    # test_loss = test_loss + 0 * binomial_loss(output, output_lengths, target_lengths)
                    num_correct += calc_acc(output, target_padded, target_lengths)
                    num_chars += sum(target_lengths)
        print('\nEpoch {} Test\t\t\tLoss: {:.4f}\tAcc: ({}/{}) {:.0f}%'.format(
            self.cur_epoch,
            test_loss.item() / num_test_batches,
            num_correct,
            num_chars,
            100. * num_correct / num_chars
        ))

        (input_padded, target_padded, output_lengths, target_lengths) = next(iter(self.test_loader))
        ex_index = randint(0, self.batch_size_test - 1)
        for x in range(target_lengths[ex_index]):
            char = target_padded[ex_index, x].item()
            if char >= 65:  # grave accent (65) should have been replaced with acute accent (8)
                char += 1
            print(chr(31 + char), end="")
        print('\n')
        temp = torch.argmax(output, dim=2)[ex_index]
        for (i, x) in enumerate(temp):
            if i < output_lengths[ex_index]:
                if x >= 65:  # grave accent (65) should have been replaced with acute accent (8)
                    x += 1
                print(chr(31 + x), end="")

        if synflow:
            print('\nSynflow Metrics: {:.4f}, {:.4f}, {:.4f}'.format(syn[0], syn[1], syn[2]))
            # dot_prod, corr, corr * n_params
        print('\n')
