import torch, sys
c = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print(f"step={c['step']}, best={c['best_rel_err']:.4e}")
print('linear' if 'bias_U' in c['model'] else 'gelu')
# check latent_dim
s = c['model']
if 'head_U.4.weight' in s:
    print(f"N={s['head_U.4.weight'].shape[0]//9}, latent={s['head_U.0.weight'].shape[1]}, ew={s['encoder.0.weight'].shape[0]}")
elif 'head_U.weight' in s:
    print(f"N={s['head_U.weight'].shape[0]//9}, latent={s['head_U.weight'].shape[1]}, ew={s['encoder.0.weight'].shape[0]}")
