"""Client/profile measurement parameters. No Backbone or task optimizer access."""
import torch
from torch import nn
from torch.nn import functional as F
from adapterfl_methods import isolated_evaluation_rng
from .config import seed_of


class Instrument(nn.Module):
    def __init__(self, inputs, classes, dim, head_seed, client, profile, state=None):
        super().__init__()
        # A fork covers CPU and all initialized CUDA RNG streams, including module init.
        with isolated_evaluation_rng():
            torch.manual_seed(head_seed)
            self.head = nn.Linear(dim, dim, bias=False)
            self.decoder = nn.Linear(dim, classes, bias=False)
            nn.init.orthogonal_(self.head.weight)
            with torch.no_grad():
                self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=1))
            torch.manual_seed(seed_of('adapter-v1', head_seed, client, profile))
            self.adapter = nn.Linear(inputs, dim)
        self.head.requires_grad_(False)
        self.decoder.requires_grad_(False)
        if state is not None:
            self.adapter.load_state_dict(state)

    def forward(self, features):
        hidden = self.head(self.adapter(features.detach()))
        return F.normalize(hidden, dim=1), self.decoder(hidden)

    def candidate(self, features, labels, cfg):
        """Fit only a candidate Adapter; caller measured both states before this call."""
        opt = torch.optim.SGD(self.adapter.parameters(), lr=cfg['adapter_lr'])
        examples = 0
        for _ in range(cfg['adapter_epochs']):
            for start in range(0, len(labels), cfg['measurement_batch_size']):
                stop = start + cfg['measurement_batch_size']
                opt.zero_grad(set_to_none=True)
                _, logits = self(features[start:stop].detach())
                F.cross_entropy(logits, labels[start:stop]).backward()
                opt.step()
                examples += len(labels[start:stop])
        return {k: v.detach().cpu().clone() for k, v in self.adapter.state_dict().items()}, examples


def merge_candidates(states):
    return {k: torch.stack([s[k] for s in states]).mean(0) for k in states[0]}
