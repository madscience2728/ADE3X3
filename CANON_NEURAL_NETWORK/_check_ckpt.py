import torch
c = torch.load('checkpoints/ckpt_N19_s0.pt', map_location='cpu', weights_only=False)
print(f"step={c['step']}, best={c['best_rel_err']:.3e}")
keys = list(c['model'].keys())
has_bias_U = any('bias_U' in k for k in keys)
print('linear_heads' if has_bias_U else 'gelu_heads')
