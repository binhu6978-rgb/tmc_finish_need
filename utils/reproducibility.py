"""Opt-in numerical execution policy; no change to the FL/SGD protocol."""
import os


def configure_deterministic_execution(enabled):
    if not enabled:
        return None  # Preserve every legacy backend setting.
    import torch
    workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if workspace not in (None, ':4096:8'):
        raise ValueError('Deterministic controls require CUBLAS_WORKSPACE_CONFIG=:4096:8')
    if workspace is None and torch.cuda.is_initialized():
        raise RuntimeError('Configure deterministic execution before initializing CUDA')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return execution_policy()


def execution_policy():
    import torch
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
