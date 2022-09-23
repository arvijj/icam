import math
import torch
import torch.nn.functional as F


def probs_to_logits(x, eps=1e-3):
    return torch.log(x*(1 - eps) + eps) - torch.log(1 - x*(1 - eps))


def softmax_maxpool(x, eps=1e-3):
    x = x.clone()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    elif x.dim() != 4:
        raise ValueError('Dimension must be 3 or 4, got %d.' % (x.dim()))
    probs = F.softmax(x, dim=1)
    logits = probs_to_logits(F.adaptive_max_pool2d(probs, (1, 1)), eps=eps)
    if x.dim() == 3:
        probs = probs.squeeze(0)
        logits = logits.squeeze(0)
    return probs, logits


def label_cond(probs, label, channel_dim=1):
    eps = torch.tensor(1e-8).to(probs.device)#cuda(non_blocking=True)
    probs = probs*label
    probs = probs/torch.maximum(eps, torch.sum(probs, dim=channel_dim, keepdim=True))
    return probs


def sample_probs(probs, num_samples):
    if probs.dim() != 4:
        raise ValueError('Dimension of prob must 4, got %d.' % (probs.dim()))
    n, c, h, w = probs.size()
    probs = probs.view(n*c, h*w)
    indices = torch.multinomial(probs, num_samples=num_samples, replacement=True)
    samples = torch.gather(probs, 1, indices)
    samples = samples.view(n, c, num_samples)
    return samples


def feature_similarity_loss(x, y, mu, sigma=None, px=1.0, py=2.0, fast=False, max_dist=10):
    """
    Computes the feature similarity loss between features x and predictions y.
    The tensors x and y must have shapes BxC1xHxW and BxC2xHxW respectively.
    """
    
    assert len(x.size()) == len(y.size()) == 4, 'Both x and y must have 4 dimensions, but they had %d and %d.' % (len(x.size()), len(y.size()))
    assert x.size()[0] == y.size()[0], 'x and y must have the same batch dimensions.'
    assert x.size()[2:] == y.size()[2:], 'x and y must have the same spatial dimensions.'

    # Format input dimensions
    n, cx, h, w = x.size()
    x = x.clone().view(n, -1, h*w).transpose(1, 2).contiguous().clamp(0., 1.)
    y = y.clone().view(n, -1, h*w).transpose(1, 2).contiguous()

    # Compute spatial distances
    if sigma is not None:
        h_range = torch.arange(0, h).to(x.device)
        w_range = torch.arange(0, w).to(x.device)
        hh, ww = torch.meshgrid(h_range, w_range)
        hh, ww = (hh.type(torch.float), ww.type(torch.float))
        pos = torch.cat([hh.unsqueeze(-1), ww.unsqueeze(-1)], dim=-1).view(1, -1, 2)
        dp = torch.cdist(pos, pos, p=2.0)

    # Compute feature (dx) and prediction (dy) distances
    dx = torch.cdist(x, x, p=px)
    dy = torch.cdist(y, y, p=py)

    # Ignore distant pixel pairs
    if fast and sigma is not None:
        dp = torch.where(dp > max_dist, torch.tensor(0.).to(dp.device), dp)
        indices = torch.nonzero(dp).T
        dx = dx[indices[0], indices[1], indices[2]]
        dy = dy[indices[0], indices[1], indices[2]]
        dp = dp[indices[0], indices[1], indices[2]]

    # Compute the loss
    dx = dx / cx
    dx = dx.clamp(0., 1.)
    dx = torch.tanh(mu + torch.log((dx + 1e-5) / (1 - dx + 1e-5)))
    dy = dy**2
    if sigma is not None:
        dp = torch.exp(-0.5*(dp/sigma)**2) / (2.*math.pi*sigma**2)
    else:
        dp = 1 / (h*w)
    fs_loss = -(1/(n*h*w)) * dp * dy * dx
    return torch.sum(fs_loss)
