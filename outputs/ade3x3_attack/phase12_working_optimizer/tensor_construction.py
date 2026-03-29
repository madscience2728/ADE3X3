from __future__ import annotations

from pathlib import Path

from cp_als import build_cp_tensor_cab, cp_loss_stats, exact_factor_matrices


OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    tensor_cab = build_cp_tensor_cab()
    c_factor, a_factor, b_factor, orientation, loader_residual = exact_factor_matrices()
    exact_loss, exact_max_abs = cp_loss_stats(tensor_cab, c_factor, a_factor, b_factor)
    payload = {
        'orientation': orientation,
        'loader_residual': loader_residual,
        'shape': list(tensor_cab.shape),
        'nonzero_count': int((tensor_cab != 0).sum()),
        'slice_nonzero_counts': [int((tensor_cab[idx] != 0).sum()) for idx in range(tensor_cab.shape[0])],
        'mode0_rank': int(__import__('numpy').linalg.matrix_rank(tensor_cab.reshape(9, 81))),
        'exact_reconstruction_loss': exact_loss,
        'exact_reconstruction_max_abs': exact_max_abs,
    }
    (OUT_DIR / 'tensor_construction_summary.json').write_text(__import__('json').dumps(payload, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()