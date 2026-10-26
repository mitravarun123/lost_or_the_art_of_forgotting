import os
os.environ.setdefault("LOI_FIGURE_DPI", "100")  # Only software fixture exports.
import torch
# Small tensor tests are faster and more reproducible with one CPU worker.
torch.set_num_threads(1)
