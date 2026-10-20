"""Run compiled PHNet's full batch forward/backward before a 1000-epoch job."""
import time

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_PHNetOriginal_1000ep import (
    PHNetDiceCE, PHNetMRIOriginal,
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Requires a CUDA GPU')
    dev = torch.device('cuda:0')
    model = PHNetMRIOriginal().to(dev).train()
    model = torch.compile(model)
    criterion = PHNetDiceCE().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)
    scaler = torch.amp.GradScaler('cuda')
    data = torch.randn((2, 1, 48, 192, 192), device=dev)
    target = torch.randint(0, 63, (2, 1, 48, 192, 192), device=dev)
    print('Compiling full-patch training step; the first step may take several minutes.', flush=True)
    torch.cuda.reset_peak_memory_stats(dev)
    for step in range(2):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            logits = model(data)
            loss = criterion(logits, target)
        assert tuple(logits.shape) == (2, 63, 48, 192, 192), logits.shape
        assert torch.isfinite(loss).item(), loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(dev)
        print(f'Step {step}: shape={tuple(logits.shape)}, '
              f'loss={loss.detach().item():.5f}, '
              f'seconds={time.perf_counter()-start:.2f}, '
              f'peak_allocated_GiB={torch.cuda.max_memory_allocated(dev)/2**30:.2f}, '
              f'peak_reserved_GiB={torch.cuda.max_memory_reserved(dev)/2**30:.2f}',
              flush=True)
    print('PASS: compiled full-patch forward/backward; random data do not test convergence.')


if __name__ == '__main__':
    main()
