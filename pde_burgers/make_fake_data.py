"""Write a small synthetic HDF5 with the PDEBench 1D Burgers layout, for local smoke tests only.

Layout mimicked: 'tensor' [N, 201, 1024], 'x-coordinate' [1024], 't-coordinate' [202].
Dynamics: random Fourier modes advected and diffused (NOT Burgers; just learnable).
"""
import argparse

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/fake/1D_Burgers_Sols_Nu0.01.hdf5")
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.default_rng(0)
    nx, nt = 1024, 201
    x = (np.arange(nx) + 0.5) / nx
    t = np.linspace(0, 2.01, nt + 1)
    k = np.arange(1, 5)
    amp = rng.normal(size=(args.n, 4)) / k
    ph = rng.uniform(0, 2 * np.pi, size=(args.n, 4))
    u = np.zeros((args.n, nt, nx), np.float32)
    for j, tj in enumerate(t[:nt]):
        arg = 2 * np.pi * k[None, :, None] * (x[None, None, :] - 0.3 * tj) + ph[:, :, None]
        u[:, j] = (amp[:, :, None] * np.exp(-0.05 * k[None, :, None] ** 2 * tj) * np.sin(arg)).sum(1)
    with h5py.File(args.out, "w") as f:
        f["tensor"] = u
        f["x-coordinate"] = x.astype(np.float32)
        f["t-coordinate"] = t.astype(np.float32)
    print(args.out)


if __name__ == "__main__":
    main()
