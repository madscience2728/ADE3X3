from torchviz import make_dot
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import KethVaraiMachine

model = KethVaraiMachine(N=19, latent_dim=1024, encoder_depth=64)
A = torch.randn(1, 9)
B = torch.randn(1, 9)
out = model(A, B)
make_dot(out, params=dict(model.named_parameters())).render("keth_varai", format="png", cleanup=True)
print("Saved -> keth_varai.png")
