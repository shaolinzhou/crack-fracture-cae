# DAT-driven crack solver (general Q4)

Generalised, DAT(GiD)-driven version of the Brazilian-disc damage solver:
reads an arbitrary unstructured four-node (Q4) mesh plus multi-material data
and runs the same two-phase (warmup → NN-coupled) scale-invariant damage
evolution. Write-only GiD post-processing files (`.msh`/`.res`).

## Expected DAT sections

- `coordinates`: `node_id x y`
- `Element`: `element_id n1 n2 n3 n4 material_id`
- `Moment-Load`: nodal freedom flags. `0` means fixed, `1` means free.
- `Presure`: nodal prescribed vertical displacement. The value is used directly.
- `Wall`: crack/wall node ids used to initialize damage.
- `MATERIAL PROPERTIES`: `mat_id E nu sigma_t K_Ic density`

## Run

From the repository root:

```powershell
python FEA/run_fea.py                     # defaults to data/c1.dat
python FEA/run_fea.py data/c1.dat --warmup 10 --coupled 0
python FEA/run_fea.py data/d1.dat         # second example mesh
```

Outputs are written to `FEA/results/snapshots` (git-ignored, regenerable).
GiD post-processing files are written next to the snapshots:

```text
FEA/results/<dat_name>.msh
FEA/results/<dat_name>.res
```

The `.res` file contains nodal/elemental displacement, stress, strain, damage,
Von Mises stress, scale exponent d, and reaction-load history.

For a different model, put the DAT file anywhere and pass its path:

```powershell
python FEA/run_fea.py path\to\your_model.dat --gid-name your_name
```

## Notes

- `numpy` and `scipy` are required. `torch` enables the NN-coupled correction;
  without it the coupled phase falls back to an analytical scale field
  `d = -0.5 + 0.3·ln(1-D)`. `matplotlib` enables PNG snapshots; without it the
  solver still writes `load_displacement.csv`.
- The solver automatically fixes one bottom-center `UX` degree of freedom if the
  DAT file has no fixed `UX` (removes rigid-body horizontal motion, `--no-auto-anchor-x`
  to disable).
- `carck.gid/` contains the GiD problem-type project used to produce the DAT
  example inputs.
