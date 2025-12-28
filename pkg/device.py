import torch


def _device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    return "cpu"


device = _device()


def get_device():
    print("device: ", device)
    return device
