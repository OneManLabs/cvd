import torch


def get_available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            devices.append(f"cuda:{i}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_display_name(device: str) -> str:
    if device.startswith("cuda"):
        idx = int(device.split(":")[1]) if ":" in device else 0
        try:
            return f"CUDA - {torch.cuda.get_device_name(idx)}"
        except Exception:
            return f"CUDA ({device})"
    if device == "mps":
        return "Apple Metal (MPS)"
    return "CPU"
