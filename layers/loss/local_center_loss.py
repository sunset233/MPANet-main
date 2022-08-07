import torch
from torch import nn


class GeMP(nn.Module):
  def __init__(self, p=3.0, eps=1e-12):
    super(GeMP, self).__init__()
    self.p = p
    self.eps = eps

  def forward(self, x):
    p, eps = self.p, self.eps
    if x.ndim != 2:
      batch_size, fdim = x.shape[:2]
      x = x.view(batch_size, fdim, -1)
    return (torch.mean(x**p, dim=-1) + eps)**(1/p)


class CrossEntropyLabelSmooth(nn.Module):
  """Cross entropy loss with label smoothing regularizer.
  Reference:
  Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
  Equation: y = (1 - epsilon) * y + epsilon / K.
  Args:
      num_classes (int): number of classes.
      epsilon (float): weight.
  """

  def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.use_gpu = use_gpu
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    """
    Args:
        inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        targets: ground truth labels with shape (num_classes)
    """
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
    if self.use_gpu: targets = targets.cuda()
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (- targets * log_probs).mean(0).sum()
    return loss.mean()


class CenterTripletLoss(nn.Module):
    def __init__(self, k_size, margin=0):
        super(CenterTripletLoss, self).__init__()
        self.margin = margin
        self.k_size = k_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Come to centers
        centers = []
        for i in range(n):
            centers.append(inputs[targets == targets[i]].mean(0))
        centers = torch.stack(centers)
                
        dist_pc = (inputs - centers)**2
        dist_pc = dist_pc.sum(1)
        dist_pc = dist_pc.sqrt()

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, centers, centers.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_an, dist_ap = [], []
        for i in range(0, n, self.k_size):
            dist_an.append( (self.margin - dist[i][mask[i] == 0]).clamp(min=0.0).mean() )
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        loss = dist_pc.mean() + dist_an.mean()
        return loss, dist_pc.mean(), dist_an.mean()
