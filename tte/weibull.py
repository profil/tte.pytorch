import torch


def weibull_loglikelihood(alpha, beta, y, u, lengths, epsilon=1e-25):
    hazard0 = ((y + epsilon) / alpha) ** beta
    hazard1 = ((y + 1) / alpha) ** beta
    hazards = torch.clamp(hazard1 - hazard0, min=1e-5, max=30)
    loglikelihoods = u * torch.log(torch.exp(hazards) - 1) - hazard1

    # TODO:
    # Make sure that the masking below is correct

    max_length, batch_size, *trailing_dims = loglikelihoods.size()

    ranges = loglikelihoods.data.new(max_length)
    ranges = torch.arange(max_length, out=ranges)
    ranges = ranges.unsqueeze_(1).expand(-1, batch_size)

    lengths = loglikelihoods.data.new(lengths)
    lengths = lengths.unsqueeze_(0).expand_as(ranges)

    mask = ranges < lengths
    mask = mask.unsqueeze_(-1).expand_as(loglikelihoods)

    return -1 * torch.mean(loglikelihoods * mask.float())
