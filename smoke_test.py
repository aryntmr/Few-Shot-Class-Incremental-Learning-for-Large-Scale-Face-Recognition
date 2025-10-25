import torch

def smoke_test():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        try:
            # Attempt to create a tensor on the GPU
            tensor = torch.tensor([1, 2, 3], device=device)
            print(f"Tensor on device {device}: {tensor}")
            return True
        except Exception as e:
            print(f"Failed to create tensor on device {device}: {e}")
            return False
    else:
        print("CUDA is not available.")
        return False

success = smoke_test()
print(f"Smoke test successful: {success}")
