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

The main entry points are listed below. Generated CSV/JSON tables, manifests,
diagnostics, and rendered figures are written outside the repository. Before
running any generator, choose an external result directory:

```bash
export ESP_ERROR_BENCH_OUTPUT_DIR=/path/outside/Ewald-Splitting-results
```

The Figure 5 artifacts will then be under
`$ESP_ERROR_BENCH_OUTPUT_DIR/redesigned_section5/`. The repository contains
only source code and required simulation inputs.

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

The lower AD row uses a target-conditioned structure-factor theory. A
dedicated prefix-only stage reads coordinate frames 1--25 and freezes the
mean and five block means of \(S_{\rm tag}(\mathbf q)\), ordinary \(S_q\) for
diagnosis, and the charge-class conditional pair amplitude
\(\mu_a(\mathbf q)\). Candidate prediction then reopens neither molecular
coordinates nor force arrays. It decomposes
\(S_{\rm tag}=S_{\rm fluct}+S_{\rm coherent}\), contracts the zero-mean part
with the exact homogeneous AD cell-moment weights, and forms a coherent
all-source pair field that includes \(i=j\). The production self-correction
vector is added before cell squaring:
\[
\Delta F_{\rm mesh}^2=\Delta F_{\rm pair,fluct}^2+
\left\langle\left|\mathbf F_{\rm pair,all,coherent}+
\mathbf F_{\rm self,correction}\right|^2\right\rangle_{\rm cell},
\qquad
\Delta F_{\rm pred}^2=\Delta F_{\rm mesh}^2+
\Delta F_{\rm Fourier}^2.
\]
The code also constructs and checks the algebraically equivalent
\(\mathbf F_{j\ne i}+\mathbf F_{\rm residual-self}\) decomposition. Thus the
pair/self cross term is retained explicitly without double counting the raw
mesh self response. The remaining approximations are the diagonal
physical-mode closure for the zero-mean pair fluctuation, omission of
coherent source aliases beyond the six nearest faces, and omission of
in-band/Fourier-tail covariance. Every prediction row and manifest records
these choices.
The pilot coordinates enter only through the frozen structure spectrum: the
formal prediction never evaluates a particlewise mesh force, finite-band
force, or their difference. Such force-operator calculations are confined to
the post-freeze diagnostic command below.
The two-harmonic self-correction coefficients are reproduced directly from
the deterministic sums in `ESP::compute_sf_precoeff()` and
`ESP::compute_gf_ad()`; the formal prediction does not read single-charge
LAMMPS force dumps. Independent unit-charge probes remain operator tests.

The fixed-influence ik prediction likewise uses unbuffered prefix readers for
only frames 1--25 of both the coordinate trajectory and the coarse PPPM force
dump. The frozen JSON records the two exact input-prefix SHA-256 digests; it
does not hash or read the remaining holdout records before selection.

Freeze the pilot spectrum first, run the estimator checks, generate the full
fixed-band curves, and freeze both measured-\(S_{\rm tag}\) selections:

```bash
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" \
  --freeze-pilot-spectrum
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" --self-test
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" --baseline
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" \
  --select-target 1e-4
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" \
  --select-target 1e-5
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
  --validate-selection 1e-4 --lmp "$LMP"
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" \
  --validate-selection 1e-5 --lmp "$LMP"
```

Run the alias-shell convergence audit and optional joint-operator/direct-band
diagnostic only after freezing. Neither output calibrates or alters the
selector. The diagnostic evaluates particlewise operators and is never a
Figure 5 prediction curve or screening input.

```bash
for TARGET in 1e-4 1e-5; do
  python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" \
    --spectral-theory-audit "$TARGET"
  python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" \
    --diagnostic-direct-check "$TARGET"
done
```

The AD candidate tables report the five-block spectrum SEM, alias
importance-sampling SEM, their quadrature combination, and a one-sided 95%
upper value using Student t with four degrees of freedom. Setting
\(S_{\rm tag}=1\) recovers the exact homogeneous all-alias AD cell-moment
estimator. Neither the raw mesh self response nor the self-correction field is
structure weighted; their cell-wise cross terms with the coherent pair field
are retained.
Frozen AD selections and their later validation remain in the external result
tables and manifests but are not plotted separately.

`lammps_ad_total_validation/run_water_ad_legacy_direct_band_diagnostic.py`
is retained only as an explicitly acknowledged finite-band diagnostic. It
requires `--legacy-direct-band-diagnostic` and is not part of the Figure 5
selection or validation workflow. Formal production ESP-AD/Ewald helpers are
in `water_ad_production.py`.

`fig5_ad_theory_common.py` contains only the fixed candidate definitions,
pilot-only normalization, and converged residual-self quadrature shared by the
current Figure 5 AD workflow. It does not implement a separate selector.
`ad_pair_self_theory.py` implements the charge-class conditional-mean
decomposition and the cell-wise pair/self vector sum. Its FINUFFT path uses an
exact frequency-shift identity to process distant alias modes in bounded
128-mode-wide tiles; this limits plan memory without truncating the requested
mode set.

Each external Figure 5 `stag_*` directory contains the pre-validation
candidate CSV, its frozen-selection JSON, pilot-block and alias audits,
independent holdout tables, and a manifest linking every generated artifact
by SHA-256.

A candidate is eligible only when `sigma_up >= 1` and its one-sided upper
value is no larger than the target. The deterministic tie-break minimizes
`M^3`, then `P`, then `c_spread`.

Each LAMMPS runner records the actual executable SHA-256 in its manifest. To
require a specific archived build, add `--require-lmp-sha256 SHA256`; the manuscript
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
files under `$ESP_ERROR_BENCH_OUTPUT_DIR/redesigned_section5/`.
To regenerate only Figure 5 after its three source tables are available, use:

```bash
python3 "$FIGDIR/plot_redesigned_main_figures.py" --figure 5
```

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
    path
    for item in listed.split(b"\0")
    if item
    for path in [Path(os.fsdecode(item))]
    if path.is_file()
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
