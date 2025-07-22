import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

def get_criterion(loss_function, lossparams):
    if loss_function == 'jsd':
        criterion, robust_samples = JsdCrossEntropy(**lossparams), lossparams["num_splits"] - 1
    elif loss_function == 'trades':
        criterion, robust_samples = Trades(**lossparams), 0
    else:
        criterion, robust_samples = torch.nn.CrossEntropyLoss(label_smoothing=lossparams["smoothing"]), 0
    test_criterion = torch.nn.CrossEntropyLoss()
    return criterion, test_criterion, robust_samples

class Criterion(nn.Module):
    def __init__(self, standard_loss, trades_loss=False, beta=1.0, step_size=0.003, epsilon=0.031,
                 perturb_steps=10, distance='l_inf', robust_loss=False, alpha=12, num_splits=3, **kwargs):
        super().__init__()
        loss = getattr(torch.nn, standard_loss)
        self.standard_criterion = loss(**kwargs)
        self.robust_samples = num_splits - 1 if robust_loss == True else 0
        if trades_loss == True:
            self.trades_criterion = Trades(step_size=step_size, epsilon=epsilon, perturb_steps=perturb_steps, beta=beta,
                                           distance=distance)
        else:
            self.trades_criterion = None
        if robust_loss == True and num_splits == 3:
            self.robust_criterion = JsdCrossEntropy(num_splits=num_splits, alpha=alpha)
        elif robust_loss == True and num_splits == 2:
            self.robust_criterion = JsdCrossEntropy(num_splits=num_splits, alpha=alpha)
        else:
            self.robust_criterion = None

    def __call__(self, outputs, mixed_targets, inputs=None, targets=None):
        split_size = outputs.shape[0] // (self.robust_samples+1)
        loss = self.standard_criterion(outputs[:split_size], mixed_targets)
        if self.trades_criterion is not None:
            if inputs == None or targets == None:
                print('no original inputs/target given for Trades loss calculation')
            trades_loss = self.trades_criterion(inputs[:split_size], targets)
            loss += trades_loss
        if self.robust_criterion is not None:
            loss += self.robust_criterion(outputs, mixed_targets)

        return loss

    def test(self, outputs, mixed_targets):
        loss = self.standard_criterion(outputs, mixed_targets)
        return loss

    def update(self, model, optimizer):
        if self.trades_criterion is not None:
            self.trades_criterion.update(model, optimizer)

class Trades(nn.Module):
    """
    TRADES loss for training adversarially robust models
    based on https://github.com/yaodongyu/TRADES built into class
    """

    def __init__(self, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, distance='l_inf'):
        super().__init__()
        self.model = None
        self.optimizer = None
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.distance = distance

    def update(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def __call__(self, x_natural, y):
        # define KL-loss
        criterion_kl = nn.KLDivLoss(size_average=False)

        self.model.eval()
        batch_size = len(x_natural)
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        p_natural = F.softmax(self.model(x_natural), dim=1)

        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(self.model(x_adv), dim=1),
                                           p_natural)
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif self.distance == 'l_2':
            delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr=self.epsilon / self.perturb_steps * 2)

            for _ in range(self.perturb_steps):
                adv = x_natural + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * criterion_kl(F.log_softmax(self.model(adv), dim=1),
                                           p_natural)
                loss.backward(retain_graph=True)
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()

                # projection
                delta.data.add_(x_natural)
                delta.data.clamp_(0, 1).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=self.epsilon)
            x_adv = Variable(x_natural + delta, requires_grad=False)
        else:
            raise ValueError(f'Attack={self.distance} not supported for TRADES training!')

        self.model.train()
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # zero gradient
        self.optimizer.zero_grad()
        # calculate robust loss

        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(self.model(x_adv)[0], dim=1),
                                                    p_natural)
        loss_robust = self.beta * loss_robust
        return loss_robust

class JsdCrossEntropy(nn.Module):
    """ Jensen-Shannon Divergence + Cross-Entropy Loss

    Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781

    Hacked together by / Copyright 2020 Ross Wightman
    """
    def __init__(self, num_splits=3, alpha=12):
        super().__init__()
        self.num_splits = num_splits
        self.alpha = alpha

    def __call__(self, output, target):
        split_size = output.shape[0] // self.num_splits
        assert split_size * self.num_splits == output.shape[0]
        logits_split = torch.split(output, split_size)

        # Cross-entropy is only computed on clean images
        probs = [F.softmax(logits, dim=1) for logits in logits_split]

        # Clamp mixture distribution to avoid exploding KL divergence
        logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-7, 1).log()
        loss = self.alpha * sum([F.kl_div(
            logp_mixture, p_split, reduction='batchmean') for p_split in probs]) / self.num_splits
        return loss


def trades_loss(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                attack='l_inf'):
    """
    TRADES training (Zhang et al, 2019).
    """

    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    logits_natural = model(x_natural)
    p_natural = F.softmax(logits_natural, dim=1)

    if attack == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    elif attack == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
            loss.backward(retain_graph=True)
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()
    # calculate robust loss
    logits_adv = model(x_adv)
    loss_natural = F.cross_entropy(logits_natural, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits_natural, dim=1))
    loss = loss_natural + beta * loss_robust

    return loss