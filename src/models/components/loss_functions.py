import torch
import torch.nn.functional as F

class DiscretizedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_bins):
        super(DiscretizedCrossEntropyLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, inputs, targets):
        # Discretize the targets into bins
        targets_bins = (targets * self.num_bins).long()
        loss = F.cross_entropy(inputs, targets_bins)
        return loss
    
class BetaDistLoss(torch.nn.Module):
    def __init__(self):
        super(BetaDistLoss, self).__init__()

    def forward(self, inputs, target):
        # Discretize the targets into bins
        shrink = 0.000001  # To avoid NaNs
        target = target * (1 - 2 * shrink) + shrink / 2

        # Get alpha and beta parameters
        alpha = inputs[:, 0]+ torch.finfo(inputs[:, 0].dtype).eps#torch.ones_like(inputs[:, 0])*shrink#/10000000
        beta = inputs[:, 1]+ torch.finfo(inputs[:, 1].dtype).eps#torch.ones_like(inputs[:, 1])*shrink#/10000000

        # Create a Beta distribution with the predicted alpha and beta parameters
        beta_dist = torch.distributions.Beta(alpha, beta)

        # Calculate the log probability of the truth value
        log_prob = beta_dist.log_prob(target) #.squeeze(1)
        return -log_prob.mean()
