# ESP error-benchmark regeneration bundle

This directory contains the source code, fixed configurations, trajectories,
reference forces, LAMMPS patch, and representative LAMMPS inputs used to
regenerate the theoretical screens and independent validation data for
manuscript Figures 2-6.

All commands below assume that this `error_bench/` directory is the current
working directory. LAMMPS input paths are relative to this directory.

## Integrity check

`SHA256SUMS` covers every distributed file except itself and macOS
`.DS_Store` metadata:

```bash
python3 - <<'PY'
import hashlib
from pathlib import Path

failed = []
for entry in Path("SHA256SUMS").read_text(encoding="utf-8").splitlines():
    expected, relative_path = entry.split("  ", 1)
    actual = hashlib.sha256(Path(relative_path).read_bytes()).hexdigest()
    status = "OK" if actual == expected else "FAILED"
    print(f"{relative_path}: {status}")
    if actual != expected:
        failed.append(relative_path)
if failed:
    raise SystemExit(f"checksum mismatch: {', '.join(failed)}")
PY
```

## Directory layout

```text
error_bench/
|-- README.md
|-- SHA256SUMS
|-- LICENSE
|-- requirements.txt
|-- lammps_esp_ik_ad_complete.patch
|-- inputs/
|   `-- representative_lammps/
|-- numerical_examples/
|   |-- random_charges/
|   |-- inhomogeneous_charges/
|   |-- water_trajectory_benchmark/
|   `-- large_water_window_upsampling/
`-- src/
    `-- numerical_test/
        |-- dump_pswf_coeff.cpp
        |-- lammps_math_pswf/
        `-- redesigned_section5/
```

- `src/numerical_test/redesigned_section5/` contains the estimators,
  data-generation runners, and plotting driver.
- `numerical_examples/random_charges/` and
  `numerical_examples/inhomogeneous_charges/` contain ten fixed 512-charge
  configurations used by Figures 2-4.
- `numerical_examples/water_trajectory_benchmark/` contains the 51-frame
  SPC/E trajectory, topology, converged Ewald force reference, and a coarse
  PPPM mesh-20 force evaluation used only to normalize the Figure 5 relative
  theoretical screen.  The latter is not an Ewald reference.
- `numerical_examples/large_water_window_upsampling/` contains the 21,624-atom
  trajectory, five Ewald reference frames, and the Figure 6 scan runners.
- `inputs/representative_lammps/` contains representative IK, AD, ESP,
  B-spline, and PPPM inputs. Run them from the bundle root.

## Python environment

Python 3.11 is recommended. Install the pinned numerical and plotting
dependencies in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

A C++17 compiler is also required. `fixed_ik_reference.py` automatically
builds `src/numerical_test/redesigned_section5/dump_pswf_coeff` from the
included C++ and MathPSWF sources when needed.

## Rebuild the patched LAMMPS executable

The combined patch contains the ESP analytical-differentiation (AD) and
`ik`-differentiation implementations. It applies to this upstream revision:

```text
repository: https://github.com/lammps/lammps.git
commit:     c4fe7a5bcf91f6b1ee634b1f79c55d671fa0badf
patch SHA-256: 218c12f5e6a4d8a0a07300aec9d9fc9d50df1fd14cd2d88a8ab46f5310a3bced
```

Clone and build beside `error_bench/`:

```bash
git clone https://github.com/lammps/lammps.git ../lammps-esp-repro
git -C ../lammps-esp-repro checkout --detach \
  c4fe7a5bcf91f6b1ee634b1f79c55d671fa0badf
git -C ../lammps-esp-repro apply --check \
  ../error_bench/lammps_esp_ik_ad_complete.patch
git -C ../lammps-esp-repro apply \
  ../error_bench/lammps_esp_ik_ad_complete.patch

cmake -S ../lammps-esp-repro/cmake -B ../lammps-esp-build \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_MPI=OFF \
  -D PKG_KSPACE=ON \
  -D PKG_MOLECULE=ON \
  -D PKG_RIGID=ON \
  -D FFT=KISS \
  -D FFT_SINGLE=OFF
cmake --build ../lammps-esp-build -j 4
```

The recorded validation configuration used AppleClang 21 with C++17,
`-O3 -DNDEBUG`, MPI stubs, double-precision KISS FFTs, and the KSPACE,
MOLECULE, and RIGID packages. MPI or FFTW builds are valid functional builds
but are not bitwise identical to that configuration.

Set the executable path for the commands below:

```bash
LMP="../lammps-esp-build/lmp"
FIGDIR="src/numerical_test/redesigned_section5"
export ESP_LAMMPS_BIN="$LMP"
```

For reproducible LAMMPS inputs, select the differentiation mode explicitly:

```lammps
kspace_modify diff ik
# or
kspace_modify diff ad
```

Both modes are available with `kspace_style esp` and the controlled
`kspace_style esp/bspline` window used by the Figure 6 comparison.

## Regenerate source data

The main entry points are listed below. They write generated CSV and JSON
files beside the corresponding runner unless stated otherwise.

### Figures 2-4

Run the Python estimators in dependency order:

```bash
python3 "$FIGDIR/run_fig2_fourier_validation.py"
python3 "$FIGDIR/run_fig2_slab_transfer.py"
python3 "$FIGDIR/run_fig2_water_fourier_prediction.py"
python3 "$FIGDIR/run_fig2_water_fourier_reference.py"
python3 "$FIGDIR/run_fig2_water_fourier_tail.py"

python3 "$FIGDIR/run_fig3_mesh_validation.py"

python3 "$FIGDIR/run_fig4_charge_spectrum.py"
python3 "$FIGDIR/run_fig4_k_resolved_contribution.py"
python3 "$FIGDIR/run_fig4_sq_correction.py"
```

Pass the patched executable explicitly to the AD operator audit and direct
Figure 3 LAMMPS validation:

```bash
python3 "$FIGDIR/ad_operator_audit/run_ad_operator_audit.py" --lmp "$LMP"
python3 "$FIGDIR/lammps_ad_total_validation/run_operator_fig3_validation.py" \
  --lmp "$LMP"
```

### Figure 5

Figure 5 separates prediction from validation. The upper fixed-influence
\(i\mathbf{k}\) row uses frozen theoretical analysis from frames 1--25,
followed by independent Ewald validation on frames 26--51:

```bash
python3 "$FIGDIR/build_fig5_fixed_ik_theory_grid.py" --stage prediction
```

The lower AD row uses target-conditioned structure factors measured from
coordinate frames 1--25. These are inserted into the exact cell-moment source
weights of the production AD operator. Residual-self and closed Fourier terms
are then added in quadrature. The prediction process reads neither holdout
coordinates nor Ewald forces.

Run the estimator checks, generate the full fixed-band curves, and freeze both
joint candidate selections:

```bash
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" --self-test
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" --baseline --lmp "$LMP"
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" \
  --joint-target 1e-4 --lmp "$LMP"
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" \
  --joint-target 1e-5 --lmp "$LMP"
```

The candidate CSV and its SHA-256 are written before each
`frozen_selection.json`. Only after those files exist, run the independent
frames-26--51 validation. The order-scan runner automatically regenerates a
case under its current `raw/` directory if a historical reused record is
absent or incomplete.

```bash
python3 "$FIGDIR/fig5_ik_ad_order_scan/run_fig5_ik_ad_order_scan.py" \
  --lmp "$LMP"
python3 "$FIGDIR/fig5_pppm_ik_ad_fixed_g_scan/run_fig5_pppm_ik_ad_fixed_g_scan.py" \
  --lmp "$LMP"
python3 "$FIGDIR/build_fig5_fixed_ik_theory_grid.py" --stage validation
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" --join-baseline-validation
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" \
  --validate-joint 1e-4 --lmp "$LMP"
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" \
  --validate-joint 1e-5 --lmp "$LMP"
```

Run the shell-convergence audit and the optional direct finite-band diagnostic
after freezing. Neither output is read by the selector:

```bash
for TARGET in 1e-4 1e-5; do
  python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" \
    --theory-audit "$TARGET"
  python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" \
    --diagnostic-direct-check "$TARGET"
done
```

The AD candidate tables report the five-block frame SEM, alias importance-
sampling SEM, their quadrature combination, and the one-sided 95% upper value
using Student t with four degrees of freedom. The theoretical total currently
neglects covariance among pair, residual-self, and Fourier contributions; this
assumption is repeated in every prediction manifest. Open black stars mark the
frozen AD selections; their later validation remains in the retained source
tables and manifests but is not plotted.

The retained Figure 5 result artifacts are under
`src/numerical_test/redesigned_section5/fig5_ad_coordinate_screen/`.
Each joint-target directory contains the pre-validation candidate CSV, its
frozen-selection JSON, pilot-block and alias audits, independent holdout
tables, and a manifest linking every retained artifact by SHA-256.

A candidate is eligible only when `sigma_up >= 1` and its one-sided upper
value is no larger than the target. The deterministic tie-break minimizes
`M^3`, then `P`, then `c_spread`.

Each runner records the actual executable SHA-256 in its manifest. To require
a specific archived build, add `--require-lmp-sha256 SHA256`; the manuscript
validation executable used SHA-256
`34332fa52c4e2ba72b9561cffbc841c9b4fdbf5809eb745b1c1656e4ac960d6a`.
Portable manifests label a configured external executable as `$LMP`; the
recorded SHA-256 identifies the binary actually used.

### Figure 6

The large-water runner accepts any patched LAMMPS executable explicitly:

```bash
python3 numerical_examples/large_water_window_upsampling/run_large_water_window_scan.py \
  --lmp "$LMP" --differentiation ik
python3 numerical_examples/large_water_window_upsampling/run_large_water_window_scan.py \
  --lmp "$LMP" --differentiation ad --output-root ad
python3 numerical_examples/large_water_window_upsampling/run_ad_pppm_p5_extension.py \
  --lmp "$LMP" --order 4 --meshes 128 135
python3 numerical_examples/large_water_window_upsampling/run_ad_pppm_p5_extension.py \
  --lmp "$LMP" --order 5 --meshes 128 135 144
```

Full Figure 5 and Figure 6 scans are computationally expensive and recreate
omitted `runs/`, force-dump, and log directories.

## Plotting

After the required source CSV files have been regenerated, the retained
plotting driver is:

```bash
python3 "$FIGDIR/plot_redesigned_main_figures.py"
```

It writes editable PDF/SVG files, 300 dpi PNG previews, and 600 dpi TIFF
files beside the plotting script.

## Representative LAMMPS inputs

Run representative inputs from the bundle root so their relative paths
resolve correctly, for example:

```bash
"$LMP" -in inputs/representative_lammps/fig6_ik_pppm_selection.in
"$LMP" -in inputs/representative_lammps/fig6_ad_pswf_selection.in
```

`fig3_ad_random_p5_m24.in` is a generated-case snapshot. Its data and
trajectory are created first by
`lammps_ad_total_validation/run_operator_fig3_validation.py`.

## Updating the checksum index

After an intentional source or input change, regenerate the index from the
bundle root:

```bash
python3 - <<'PY'
import hashlib
import os
import subprocess
from pathlib import Path

listed = subprocess.check_output(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "."]
)
paths = sorted(
    Path(os.fsdecode(item)) for item in listed.split(b"\0") if item
)
lines = []
for path in paths:
    if path != Path("SHA256SUMS"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.as_posix()}\n")
Path("SHA256SUMS").write_text("".join(lines), encoding="utf-8")
PY
```

Do not add generated `raw/` or `runtime/` directories, compiled executables,
cache files, transient checkpoints, or rendered figures to the checksum index.
