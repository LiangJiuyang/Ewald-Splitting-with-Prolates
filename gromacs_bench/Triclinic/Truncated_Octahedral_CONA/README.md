# 02_cona_abfe_truncated_octahedron

Source data: https://zenodo.org/records/12530173

This folder is prepared as a self-contained benchmark bundle for the CONA
ABFE truncated-octahedron system used in the ESP/PME comparisons. The original
Zenodo archive was reduced to the branch needed for the ligand-decoupling-from-
complex tests, then augmented with local helper scripts and pre-generated
outputs.

## Included inputs

- `system/`: extracted topology/coordinate branch
- `system/charmm36-feb2021.ff/`: force-field include tree used by the topology
- `mdp/`: original MDP files plus ESP/PME variants
- `raw/`: original downloaded archives for provenance
- `100-conf/`: 100 sampled configurations used for force-error evaluation
- `force_eval_100conf/`: per-frame force-evaluation workflow and summaries
- `timeperformance/`: MPI strong-scaling scripts, MDP variants, and collected
  timing tables
- `reproduce/`: numbered entry points for rerunning the main tests

## Software requirements

- GROMACS with ESP support for `esp.mdp` and `esp_ref.mdp`
- Standard GROMACS build for PME reference runs
- `python3` with `numpy` for postprocessing force errors
- `mpirun` for the timing scripts
- Optional Slurm for the `timeperformance/*.slurm` submissions

All launchers accept environment-variable overrides for the GROMACS binaries,
so this folder can be copied elsewhere without editing absolute paths.

## Quick start

1. Build the starting TPR:

   `./reproduce/01_prepare_em_tpr.sh`

2. Prepare `100-conf/` using either of these options:

   Generate locally:
   `./reproduce/02_generate_100conf.sh`

   Or download the pre-generated `100-conf/` folder from:
   `download_link.txt`

   After download, the package root should contain:
   `100-conf/frame_001.gro`, `100-conf/frame_002.gro`, ...

3. Rerun the 100-frame ESP/PME/PME_REF force comparison and write summary
   tables:

   `./reproduce/03_force_eval_100conf.sh`

4. Rerun one timing point of the MPI sweep, for example `np=96`:

   `./reproduce/04_timing_np_sweep.sh 96`

## Main outputs

- `cona_em.tpr`: initial TPR generated from `mdp/em.mdp`
- `100-conf/frame_*.gro`: sampled configurations after equilibration/production
- `force_eval_100conf/force_error_summary.tsv`
- `force_eval_100conf/force_error_global.txt`
- `timeperformance/mpi_forcepme_np*/`: per-run timing logs and summaries

## Notes

- `prepare_tpr.sh` now auto-detects a reasonable `gmx` binary if
  `GROMACS_BIN` is not set.
- `run_100k_equil_then_dump.sh` and the force/timing scripts already expose the
  important runtime controls via environment variables.
- Existing result folders were left in place; the `reproduce/` scripts are the
  clean entry points for rerunning the benchmark.

## Release package

- `./pack_release.sh`

This writes a cleaned release folder and a `.tar.gz` archive under `release/`,
keeping only the inputs and scripts needed to reproduce the tests. The release
package excludes `100-conf/` and instead ships `download_link.txt`.

The release package intentionally omits:

- old result directories such as `runs/`, `force_eval_100conf_archive/`, and
  collected timing outputs
- prebuilt `.tpr` files and other regenerated intermediates
- unrelated diagnostic files
