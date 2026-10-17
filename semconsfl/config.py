"""Frozen engineering defaults, not accuracy-tuned hyperparameters."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SemanticConfig:
    num_classes: int = 100
    temperature: float = 1.0
    measurement_batch_size: int = 50
    min_class_samples: int = 5
    min_error_mass: float = 1e-6
    confidence_tolerance: float = 1e-7
    anchor_decay: float = 0.9
    min_anchor_clients: int = 2
    max_age_rounds: int = 100
    profile_capacity: int = 200
    min_profile_records: int = 10
    iqr_floor: float = 1e-6
    client_capacity: int = 5
    min_client_records: int = 2
    shrinkage: float = 2.0
    gate_strength: float = 0.25

    def __post_init__(self):
        for value in self.__dict__.values():
            if not math.isfinite(value):
                raise ValueError("Nonfinite semantic configuration")
        if self.temperature != 1 or self.num_classes < 2:
            raise ValueError("This definition fixes T=1 and needs >=2 classes")
        for key in ("measurement_batch_size", "min_class_samples", "min_anchor_clients",
                    "max_age_rounds", "profile_capacity", "min_profile_records",
                    "client_capacity", "min_client_records"):
            value = getattr(self, key)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"Invalid {key}")
        if self.min_profile_records > self.profile_capacity or self.min_client_records > self.client_capacity:
            raise ValueError("History capacity is smaller than required support")
        if not (0 < self.gate_strength < 1 and 0 <= self.anchor_decay < 1):
            raise ValueError("Invalid bounded Gate or EMA")
        if not (0 < self.min_error_mass < 1 and self.iqr_floor > 0 and self.shrinkage > 0):
            raise ValueError("Invalid numerical floor/shrinkage")
        if self.confidence_tolerance < 0:
            raise ValueError("Invalid confidence tolerance")
