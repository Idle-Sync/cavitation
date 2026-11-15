"""PyTorch's ReduceLROnPlateau learning rates for fixed measure sequences.

Prints, for each case, the rate after each `step(measure)`, as the reference for the
`reduce_on_plateau_matches_pytorch` test in crates/cavitation-nn/src/train/fit.rs. Run with
`scripts/fixtures/.venv/bin/python -I scripts/fixtures/reduce_on_plateau.py`.
"""

import torch

# A loss that falls, stalls, dips slightly, stalls again and rises; and an accuracy that
# climbs, stalls and climbs again.
LOSS = [1.0, 0.8, 0.8, 0.8, 0.79995, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.9, 0.9, 0.9, 0.9]
ACCURACY = [0.5, 0.6, 0.6, 0.6, 0.6, 0.65, 0.65, 0.65, 0.65, 0.65, 0.7, 0.7, 0.7, 0.7]

CASES = [
    ("loss, defaults with patience 2", LOSS, dict(mode="min", patience=2)),
    ("loss, factor 0.5, abs 0.01, cooldown 1",
     LOSS, dict(mode="min", factor=0.5, patience=1, threshold=0.01, threshold_mode="abs",
                cooldown=1)),
    ("loss, factor 0.1, patience 0, min_lr 1e-4",
     LOSS, dict(mode="min", factor=0.1, patience=0, min_lr=1e-4)),
    ("accuracy, max, factor 0.5, patience 1", ACCURACY, dict(mode="max", factor=0.5, patience=1)),
    ("accuracy, max, abs 0.02, cooldown 2",
     ACCURACY, dict(mode="max", factor=0.5, patience=0, threshold=0.02, threshold_mode="abs",
                    cooldown=2)),
]

for name, measures, options in CASES:
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **options)
    rates = []
    for m in measures:
        scheduler.step(m)
        rates.append(optimizer.param_groups[0]["lr"])
    print(f"{name}: {[repr(r) for r in rates]}".replace("'", ""))
