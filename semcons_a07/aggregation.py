"""Frozen A07 dispatcher. Nonneutral scientific activation is deliberately locked."""


def neutral_a07(hook, uploads, lens, previous, dispatched=None, gamma=None):
    if gamma is not None and any(g != 1 for g in gamma):
        raise ValueError('No scientific Go: nonneutral semantic aggregation is not enabled')
    # Return the original result directly, without an equivalent delta reconstruction.
    return hook.aggregate(uploads, lens, previous, dispatched)
