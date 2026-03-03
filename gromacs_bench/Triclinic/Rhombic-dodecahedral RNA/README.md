# Cryo-EM-refinement

Source paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12084557/
Original upstream repo: https://github.com/ElisPo/Cryo-EM-refinement

This folder contains the cryo-EM refinement input set plus the local ESP/PME
benchmark scripts used to compare force errors and timing on the same system.
The original scientific inputs are kept alongside the additional benchmark
launchers and collected outputs.

## Included inputs

- `1-plain-md/`, `2-ss-refinement/`, `3-ensemble-refinement/`: original stages
  from the published cryo-EM-refinement workflow
- `top/`, `mdp/`, `scripts/`: topology, runtime parameters, and density-analysis
  utilities
- `equilibration/`: locally prepared equilibration workflow and outputs
- `100-conf/`: 100 sampled configurations used for force-error benchmarking
- `reproduce/`: numbered entry points for the reproducibility workflow

## Reproduce the local ESP/PME tests

1. Rebuild the equilibration trajectory from the plain-MD starting point:

   `./reproduce/01_run_full_equilibration.sh`

2. Prepare `100-conf/` using either of these options:

   Download the pre-generated `100-conf/` folder from:
   `download_link.txt`

   After download, the package root should contain:
   `100-conf/frame_001.gro`, `100-conf/frame_002.gro`, ...

   Or regenerate it locally from the included NPT restart state:
   `equilibration/run_npt_post100k_dump100.sh`

3. Regenerate the first-100-frame force comparison:

   `./reproduce/02_force_eval_first100.sh`

4. Rerun the ESP/PME parameter sweep with MPI scaling:

   `./reproduce/03_timing_param_sweep.sh`

5. Rerun the PME-only MPI scaling benchmark:

   `./reproduce/04_timing_pme_only.sh`

## Main outputs

- `equilibration/em_full.*`, `equilibration/nvt_full.*`, `equilibration/npt_full.*`
- `force_eval_first100_esp_pme_pmeref_*/`
- `timing_param_sweep_np_scaling_repeat*_npme0_*/`
- `timing_pme_only_mpi_scaling_repeat*_npme0_*/`

## Runtime notes

- The benchmark scripts now derive paths from the folder location instead of
  hardcoded `/mnt/...` roots.
- Set `GMX`, `GMX_BIN`, `GMX_ESP`, `GMX_PME`, `ROOT_DIR`, `MDP_DIR`, or
  related variables if you want to use different builds or move the dataset.
- If your environment already provides OpenMPI/CUDA/FFTW, set
  `SKIP_MODULES=1` to suppress the built-in `module load ...` calls.
- The original cryo-EM refinement inputs are kept untouched; the `reproduce/`
  launchers point only at the benchmark-related reruns.

## Release package

- `./pack_release.sh`

This writes a cleaned release folder and a `.tar.gz` archive under `release/`,
excluding the large temporary, historical, and result-heavy directories that
are not needed for reproduction. The release package excludes `100-conf/` and
instead ships `download_link.txt`.

The release package intentionally omits:

- historical timing/result directories such as `timing_*`, `scan_*`,
  `Old_Test_Results/`, and force-eval output folders
- temporary scratch directories such as `.tmp_*`
- cached files such as `__pycache__/`
