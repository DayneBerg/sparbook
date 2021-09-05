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


class OverFitter:
    def __init__(self,
                 optimizer,
                 network,
                 num_datapoints=1
                 ):
        torch.backends.cudnn.enabled = False  # forces random number generation to be deterministic
        # 761 lines of data, max 856 pix per line, max 80 chars per line
        # target accuracy is either 99.18% or 99.8856% for 0.5 errors per line or per page

        self.num_datapoints = num_datapoints
        self.epoch_size = 45 * 16
        self.log_interval = 0.25 * self.epoch_size
        self.cur_iter = 0
        self.network = network
        self.optimizer = optimizer(self.network.parameters())
        dataset = SparbookDataset()
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=num_datapoints,
            shuffle=True,  # experiment later with size-grouped sampler
            collate_fn=pad_seq,
        )
        (self.input_padded, self.target_padded, self.output_lengths, self.target_lengths) = next(iter(loader))

    def train(self, n_epochs, synflow=False):
        if self.cur_iter == 0:
            self.test_once(synflow=synflow)
        for epoch in range(n_epochs):
            for iteration in range(self.epoch_size // self.num_datapoints):
                self.cur_iter += 1
                self.train_once(self.input_padded, self.target_padded, self.output_lengths, self.target_lengths)
            print('Epoch {} '.format(epoch))
            self.test_once(synflow=synflow)

    def train_once(self, input_padded, target_padded, output_lengths, target_lengths):
        self.network.train(mode=True)
        # input should be N,W,27+4
        '''print(input_padded.shape)
        assert (len(input_padded.shape) == 3), "input dimensions were incorrect (1)"
        assert (input_padded.shape[0] == self.batch_size_train), "input dimensions were incorrect (2)"
        assert (input_padded.shape[2] == 31), "input dimensions were incorrect (3)"'''
        self.optimizer.zero_grad()
        output = F.log_softmax(self.network(input_padded), dim=2)
        # N, W/2, 95
        loss = F.ctc_loss(output.permute(1, 0, 2), target_padded, output_lengths, target_lengths)
        # loss = loss + 1 * binomial_loss(output, output_lengths, target_lengths)
        loss.backward()
        self.optimizer.declare(loss)
        self.optimizer.step()
        num_correct = calc_acc(output, target_padded, target_lengths)
        num_chars = sum(target_lengths)
        if math.floor(self.cur_iter * self.num_datapoints / self.log_interval) > \
                math.floor((self.cur_iter - 1) * self.num_datapoints / self.log_interval):
            print('Iteration {}\t\tLoss: {:.4f}\tAcc: ({}/{}) {:.0f}%'.format(
                self.cur_iter,
                loss.item(),
                num_correct,
                num_chars,
                100. * num_correct / num_chars
            ))
            print(self.optimizer.log)

    def test_once(self, synflow=False):
        num_correct = 0
        num_chars = 0
        test_loss = 0
        self.network.train(mode=False)
        self.optimizer.zero_grad()
        if synflow:
            output = F.log_softmax(self.network(self.input_padded), dim=2)
            test_loss = test_loss + F.ctc_loss(
                output.permute(1, 0, 2),
                self.target_padded,
                self.output_lengths,
                self.target_lengths
            )
            # test_loss = test_loss + 0 * binomial_loss(output, self.output_lengths, self.target_lengths)
            test_loss.backward()
            syn = calc_metric(self.network.parameters())
            num_correct += calc_acc(output, self.target_padded, self.target_lengths)
            num_chars += sum(self.target_lengths)
        else:
            with torch.no_grad():
                output = F.log_softmax(self.network(self.input_padded), dim=2)
                test_loss = test_loss + F.ctc_loss(
                    output.permute(1, 0, 2),
                    self.target_padded,
                    self.output_lengths,
                    self.target_lengths
                )
                # test_loss = test_loss + 0 * binomial_loss(output, self.output_lengths, self.target_lengths)
                num_correct += calc_acc(output, self.target_padded, self.target_lengths)
                num_chars += sum(self.target_lengths)
        print('\nIteration {} Test\t\tLoss: {:.4f}\tAcc: ({}/{}) {:.0f}%'.format(
            self.cur_iter,
            test_loss.item(),
            num_correct,
            num_chars,
            100. * num_correct / num_chars
        ))

        ex_index = randint(0, self.num_datapoints - 1)
        for x in range(self.target_lengths[ex_index]):
            char = self.target_padded[ex_index, x].item()
            if char >= 65:  # grave accent (65) should have been replaced with acute accent (8)
                char += 1
            print(chr(31 + char), end="")
        print('\n')
        temp = torch.argmax(output, dim=2)[ex_index]
        for x in temp:
            if x >= 65:  # grave accent (65) should have been replaced with acute accent (8)
                x += 1
            print(chr(31 + x), end="")

        if synflow:
            print('Synflow Metrics: {:.4f}, {:.4f}, {:.4f}'.format(syn[0], syn[1], syn[2]))
            # dot_prod, corr, corr * n_params
        print('\n')