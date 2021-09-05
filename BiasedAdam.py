import math

import torch
from torch.optim import Optimizer


class BiasedAdam(Optimizer):
    """
    Implementation of AdamW that omits bias correction for the momentum term
    This is effectively a learning-rate schedule which starts at lr * betas[0] and approaches lr

    lr (float): learning rate
    betas (float, float): parameters of exponential moving average of numerator and denominator, respectively
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), decay=0.0, power=1.0, log_estimates=False):
        if not 0.0 < lr:
            raise ValueError('Invalid learning rate: {} must be > 0.0'.format(lr))
        if len(betas) != 2 or (not 0.0 <= betas[0] < 1.0) or (not 0.0 <= betas[1] < 1.0):
            raise ValueError('Invalid betas: {} must have two elements in [0.0, 1.0)'.format(betas))
        if not 0.0 <= decay:
            raise ValueError('Invalid decay rate: {} must be >= 0.0'.format(decay))
        self.num_steps = 1
        self.log_estimates = log_estimates
        self.log = ''

        defaults = dict(lr=lr, betas=betas, decay=decay, power=power)
        super(BiasedAdam, self).__init__(params, defaults)

    def declare(self, _):
        pass

    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError('Closure is unsupported')

        log = [torch.tensor(0.0), torch.tensor(0.0)]
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('Sparse gradients are unsupported')
                    state = self.state[p]
                    if self.num_steps == 1:
                        state['numerator'] = (1 - group['betas'][0]) * grad
                        state['denominator'] = (1 - group['betas'][1]) * torch.square(grad)
                    else:
                        state['numerator'] = group['betas'][0] * state['numerator'] + (1 - group['betas'][0]) * grad
                        state['denominator'] = group['betas'][1] * state['denominator'] + \
                                               (1 - group['betas'][1]) * torch.square(grad)
                    coefficient = group['lr'] * math.sqrt(1 - math.pow(group['betas'][1], self.num_steps))
                    if group['power'] != 1.0:
                        coefficient *= math.pow(1 - math.pow(group['betas'][0], self.num_steps), group['power'] - 1)
                    updates = coefficient * state['numerator'] / torch.sqrt(1e-08 + state['denominator'])
                    if group['decay'] != 0.0:
                        updates = updates + group['decay'] * p.data
                    p.data = p.data - updates
                    if self.log_estimates:
                        log[0] = log[0] + torch.sum(torch.square(grad))
                        log[1] = log[1] + torch.sum(torch.square(updates))
        if self.log_estimates:
            self.log = '[grad_norm: {}\tupdate_norm: {}]'.format(torch.sqrt(log[0]), torch.sqrt(log[1]))
        self.num_steps += 1
