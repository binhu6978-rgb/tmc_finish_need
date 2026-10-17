"""Federated training algorithms.

Algorithms are imported from their concrete modules by the entry points.  Do
not eagerly import every baseline here: several baselines have optional
optimizer dependencies that LiquidFL does not require.
"""
