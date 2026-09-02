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
shasum -a 256 -c SHA256SUMS
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
- `numerical_examples/water_trajectory_benchmark/` contains the 50-frame
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

The AD operator audit and the direct Figure 3 LAMMPS validation expect a
LAMMPS executable at the historical runner location. A locally built binary
can be placed there for a fresh calculation:

```bash
mkdir -p "$FIGDIR/pppm_symmetric_scan"
cp "$LMP" "$FIGDIR/pppm_symmetric_scan/lmp.pppm_symmetric_scan"
python3 "$FIGDIR/ad_operator_audit/run_ad_operator_audit.py"
python3 "$FIGDIR/lammps_ad_total_validation/run_operator_fig3_validation.py"
```

### Figure 5

Figure 5 separates prediction from validation. The upper fixed-influence
\(i\mathbf{k}\) row uses frozen theoretical analysis from frames 1--25,
followed by independent Ewald validation on frames 26--50:

```bash
python3 "$FIGDIR/build_fig5_fixed_ik_theory_grid.py" --stage prediction
```

Next generate the nonoverlapping ESP and PPPM validation data, then attach
those measurements to the already frozen theoretical record:

```bash
python3 "$FIGDIR/fig5_ik_ad_order_scan/run_fig5_ik_ad_order_scan.py" \
  --lmp "$LMP"
python3 "$FIGDIR/fig5_pppm_ik_ad_fixed_g_scan/run_fig5_pppm_ik_ad_fixed_g_scan.py" \
  --lmp "$LMP"
python3 "$FIGDIR/build_fig5_fixed_ik_theory_grid.py" --stage validation
```

The lower AD row combines finite-band theoretical analysis with a 25-frame
pilot correction. Dashed curves use frames 1--25, and filled markers report
independent validation on frames 26--50.

First generate the supporting finite-band components and AD prediction tables:

```bash
python3 "$FIGDIR/build_fig5_ad_rigid_sq_theory.py" --prediction-only
python3 "$FIGDIR/run_ad_rigid_theory_selection.py" --target 1e-4 --prediction-only
python3 "$FIGDIR/run_ad_rigid_theory_selection.py" --target 1e-5 --prediction-only

python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" --baseline
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" --joint-target 1e-4
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" --joint-target 1e-5
```

Then attach the nonoverlapping frames-26--50 validation results:

```bash
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" --join-baseline-validation
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" --validate-joint 1e-4
python3 "$FIGDIR/build_fig5_ad_coordinate_screen.py" --validate-joint 1e-5
```

The black stars mark the selected AD candidates and their validation.

Each runner records the actual executable SHA-256 in its manifest. To require
a specific archived build, add `--require-lmp-sha256 SHA256`; the manuscript
validation executable used SHA-256
`34332fa52c4e2ba72b9561cffbc841c9b4fdbf5809eb745b1c1656e4ac960d6a`.

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
find . -type f ! -name SHA256SUMS ! -name .DS_Store ! -path '*/__pycache__/*' -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 shasum -a 256 > SHA256SUMS
```

Do not add generated scan directories, compiled executables, cache files, or
rendered figures to the checksum index unless the distribution policy is
changed deliberately.
