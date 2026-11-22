"""Differentiable layer mixing and the same ten feature groups used by Polos."""
import torch
from torch import nn

class CachedScalarMix(nn.Module):
    def __init__(self, normalization='sample', dropout=.1, trainable=True):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(13), requires_grad=trainable)
        self.gamma = nn.Parameter(torch.ones(()), requires_grad=trainable)
        self.normalization = normalization
        self.dropout = dropout

    def forward(self, data):
        cls, mean, variance, count = [data[k] for k in ['cls', 'mean', 'var', 'count']]
        if self.normalization == 'batch':
            weights = count.unsqueeze(-1)
            average = (mean * weights).sum(0, keepdim=True) / weights.sum(0, keepdim=True)
            variance = ((variance + (mean - average) ** 2) * weights).sum(0, keepdim=True) / weights.sum(0, keepdim=True)
            mean = average
        elif self.normalization != 'sample':
            raise ValueError(self.normalization)
        normalized = (cls - mean.unsqueeze(-1)) / (variance.unsqueeze(-1) + 1e-12).sqrt()
        shape = [1] + list(cls.shape[1:-2]) + [13, 1]
        logits = self.logits.view(*([1] * (len(shape) - 2)), 13, 1).expand(shape)
        if self.training and self.dropout:
            logits = torch.where(torch.rand(shape, device=cls.device) > self.dropout, logits,
                                 torch.full_like(logits, -1e20))
        return self.gamma * (normalized * logits.softmax(-2)).sum(-2)

class CachedPolos(nn.Module):
    def __init__(self, dimension, normalization='sample', standardize=True, mix_trainable=True,
                 head='mlp', dropout=.1):
        super().__init__()
        self.mix = CachedScalarMix(normalization, dropout if mix_trainable else 0., mix_trainable)
        self.standardize = standardize
        self.kind = head
        self.register_buffer('feature_mean', torch.zeros(dimension))
        self.register_buffer('feature_scale', torch.ones(dimension))
        if head == 'mlp':
            self.head = nn.Sequential(nn.Linear(dimension, 2304), nn.Tanh(), nn.Dropout(dropout),
                                      nn.Linear(2304, 1152), nn.Tanh(), nn.Dropout(dropout),
                                      nn.Linear(1152, 1))
        elif head in ['linear', 'ridge']:
            self.head = nn.Linear(dimension, 1)
        else:
            raise ValueError(head)

    def features(self, batch, standardize=None):
        cand = self.mix(batch['candidate'])[:, None, :]
        ref = self.mix(batch['references'])
        cand = cand.expand_as(ref)
        rc = batch['reference_vl']
        cc = batch['candidate_vl'][:, None, :].expand_as(rc)
        im = batch['image'][:, None, :].expand_as(rc)
        blocks = [ref, cand, (cand - ref).abs(), cand * ref,
                  im, cc, (im - cc).abs(), im * cc, (rc - cc).abs(), rc * cc]
        if standardize if standardize is not None else self.standardize:
            # Each group is standardized with its training-set statistics before concatenation.
            means = self.feature_mean.split([b.shape[-1] for b in blocks])
            scales = self.feature_scale.split([b.shape[-1] for b in blocks])
            blocks = [(b - mean) / scale for b, mean, scale in zip(blocks, means, scales)]
        return torch.cat(blocks, dim=-1)

    def forward(self, batch, return_features=False):
        features = self.features(batch)
        scores = self.head(features).squeeze(-1)
        if self.kind != 'ridge':
            scores = scores.sigmoid()
        prediction = scores.max(dim=1).values
        return (prediction, features) if return_features else prediction

    def load_original(self, state):
        with torch.no_grad():
            self.mix.logits.copy_(torch.cat([state[f'scalar_mix.scalar_parameters.{i}'] for i in range(13)]))
            self.mix.gamma.copy_(state['scalar_mix.gamma'].reshape(()))
            for index, layer in [(0, 1), (3, 2), (6, 3)]:
                for kind in ['weight', 'bias']:
                    getattr(self.head[index], kind).copy_(state[f'ff.0.ff.linear_{layer}.{kind}'])
        self.standardize = False
        self.mix.normalization = 'batch'
        return self

def make_model(config, stats=None):
    model = CachedPolos(6144 if config['backbone'] == 'clip' else 9984,
                        normalization=config['normalization'], standardize=config['standardize'],
                        mix_trainable=config['mix_lr'] > 0, head=config.get('head', 'mlp'))
    if stats is not None:
        model.feature_mean.copy_(stats['mean'])
        model.feature_scale.copy_(stats['scale'])
    return model
