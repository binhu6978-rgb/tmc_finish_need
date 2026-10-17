"""Read-only class-conditional probabilities with identical pre/post batches."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextmanager
def isolated_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


@dataclass
class Signature:
    probabilities: torch.Tensor  # CPU float64 [C,C], zero for absent classes
    counts: torch.Tensor         # CPU int64 [C]; distinct local examples
    metadata: dict

    def to_dict(self, min_count=1):
        # Invalid numerical rows are explicitly missing, never JSON NaN.
        return {"probabilities": {str(c): row.tolist() if torch.isfinite(row).all() else None
                                  for c, row in enumerate(self.probabilities)
                                  if int(self.counts[c]) >= min_count},
                "counts": self.counts.tolist(), "metadata": self.metadata}


def measure(model, dataset, indices, config, device):
    """The dataset must expose the protocol's clean, deterministic input view.

    Entire sorted/deduplicated local train set; no sampling or drop_last. BN
    without running statistics still uses these exact batch memberships.
    Exceptions propagate while modes, buffers and all RNG streams are restored.
    """
    indices = sorted(set(int(i) for i in indices))
    if any(i < 0 or i >= len(dataset) for i in indices):
        raise ValueError("Measurement index outside dataset")
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    c = config.num_classes
    sums = torch.zeros(c, c, dtype=torch.float64)
    counts = torch.zeros(c, dtype=torch.long)
    batch_sizes = []
    label_sequence = []
    with isolated_rng():
        try:
            model.eval()
            loader = DataLoader(Subset(dataset, indices),
                                batch_size=config.measurement_batch_size,
                                shuffle=False, drop_last=False, num_workers=0,
                                generator=torch.Generator().manual_seed(0))
            with torch.no_grad():
                for images, labels in loader:
                    labels = torch.as_tensor(labels, dtype=torch.long).cpu()
                    out = model(images.to(device))
                    logits = out["output"] if isinstance(out, dict) else out
                    if logits.shape != (len(labels), c) or ((labels < 0) | (labels >= c)).any():
                        raise ValueError("Invalid class/output shape in functional measurement")
                    probs = torch.softmax(logits / config.temperature, dim=1).double().cpu()
                    sums.index_add_(0, labels, probs)
                    counts += torch.bincount(labels, minlength=c)
                    batch_sizes.append(len(labels))
                    label_sequence.extend(labels.tolist())
        finally:
            with torch.no_grad():
                for buffer, previous in buffers:
                    buffer.copy_(previous)
            for module, training in modes:
                module.training = training
    means = sums / counts.clamp_min(1).unsqueeze(1)
    digest = hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()
    label_digest = hashlib.sha256(json.dumps(label_sequence, separators=(",", ":")).encode()).hexdigest()
    return Signature(means, counts, {"indices_sha256": digest, "batch_sizes": batch_sizes,
                     "labels_sha256": label_digest,
                     "batch_size": config.measurement_batch_size, "temperature": 1.0,
                     "order": "sorted_unique", "input_view": "clean", "num_workers": 0})


def confusion(signature, class_id, config):
    """Return (conditional wrong-class distribution, true confidence, error mass)."""
    row = signature.probabilities[class_id]
    if signature.counts[class_id] < config.min_class_samples:
        return None, "insufficient_class_samples"
    if (row.shape != (config.num_classes,) or not torch.isfinite(row).all()
            or (row < 0).any() or abs(float(row.sum()) - 1) > 1e-5):
        return None, "invalid_probabilities"
    wrong = torch.cat((row[:class_id], row[class_id + 1:]))
    mass = float(wrong.sum())
    if mass < config.min_error_mass:
        return None, "insufficient_error_mass"
    return (wrong / mass, float(row[class_id]), mass), "valid"
