# LAMMPS-ESP

LAMMPS-ESP is a fork of the [LAMMPS](https://github.com/lammps/lammps) molecular dynamics software that preserves the LAMMPS user 
interface while replacing the treatment of Coulomb interactions with ESP—an Ewald-summation variant 
based on prolate spheroidal wave functions (PSWFs). 

# Background

Molecular dynamics (MD) simulations are essential tools in materials science, computational chemistry, 
structural biology, and drug discovery. Fast Ewald summation is
the most widely used method for evaluating long-range Coulomb interactions in MD, and it
remains a performance bottleneck in all major open-source and commercial MD packages,
particularly in massively parallel simulations involving $10^9$-$10^{12}$ time steps. Notably, 
[LAMMPS](https://scholar.google.com/citations?user=Ny7N6KQAAAAJ&hl=en) ($>80,000$ citations on Google Scholar), 
[GROMACS](https://scholar.google.nl/citations?user=IHbqqNEAAAAJ&hl=nl) ($>85,000$ citations), and NAMD
all rely on Ewald-based approaches, such as the (Smooth) Particle-Mesh Ewald ([PME/SPME](https://scholar.google.com/citations?user=y4Ts-LkAAAAJ&hl=en), $>57,000$ citations)
and the Particle-Particle-Particle-Mesh ([PPPM](https://scholar.google.com/scholar?q=%22Computer+Simulation+Using+Particles%22+Hockney+Eastwood), $>12,000$ citations) methods. 

# Introduction

Without any loss of accuracy, ESP alters the fast Ewald pipeline in two places. 
First, for kernel splitting it uses PSWFs instead of Gaussians, which—thanks to the optimal concentration 
of PSWFs among band-limited functions—significantly reduces the required Fourier grid. With everything else 
equal, the FFT length drops by about a factor of two per dimension at high accuracy (≈8× in 3D). The 
residual kernel also vanishes at the real-space cutoff, eliminating any need for an “energy shift.” 
Second, for particle–mesh operations ESP employs PSWFs in place of the B-splines used by SPME. For comparable 
accuracy without k-space upsampling, PSWFs require fewer neighboring grid points (e.g., ~8 vs ~12 for five-digit accuracy). 
In contrast, native LAMMPS typically sets the spreading/interpolation order to P=5 for MPI runs and P=4 for 
single-core low-accuracy runs, which forces substantial Fourier-space upsampling. Consequently, at the same cutoff 
radius, native GROMACS often needs a much larger FFT grid—even for ≈10⁻³ force accuracy—whereas ESP achieves similar 
accuracy with far shorter transforms, yielding roughly a sixfold reduction in FFT length when other parameters are held fixed.
We implemented ESP as modular components in both LAMMPS and GROMACS, introducing only minimal, localized changes to enable 
rigorous and fair comparisons with the native codes.


# Quick Start

## Build LAMMPS-ESP
LAMMPS-ESP follows the standard LAMMPS CMake workflow. A minimal local build for ESP support is:

```bash
cmake -S LAMMPS/cmake -B build \
  -D CMAKE_BUILD_TYPE=Release \
  -D PKG_KSPACE=on \
  -D PKG_MOLECULE=on
cmake --build build -j 4
```

If you want to run the bundled SPC/E water example in [`LAMMPS-Water/`](./LAMMPS-Water), also enable `PKG_RIGID=on` because the input script uses `fix shake`.

```bash
cmake -S LAMMPS/cmake -B build \
  -D CMAKE_BUILD_TYPE=Release \
  -D PKG_KSPACE=on \
  -D PKG_MOLECULE=on \
  -D PKG_RIGID=on
cmake --build build -j 4
```

## Use ESP in LAMMPS
The current LAMMPS interface exposed by this repository is:

```lammps
kspace_style esp ACCURACY
pair_style coul/esp CUTOFF
pair_style lj/cut/coul/esp LJ_CUTOFF
```

For example, to replace the standard long-range Coulomb treatment

```lammps
pair_style lj/cut/coul/long 9.0 9.0
kspace_style ewald 1.0e-6
```

with ESP, use

```lammps
pair_style lj/cut/coul/esp 9.0
kspace_style esp 1.0e-6
```

For Coulomb-only simulations, use

```lammps
pair_style coul/esp 8.0
kspace_style esp 1.0e-7
```

You can still control the Fourier grid and interpolation order through `kspace_modify`, for example:

```lammps
kspace_modify mesh 100 100 100 order 7
```

Note that the `esp` accuracy argument is a target parameter used by the solver setup. The realized force/energy error depends on the system, cutoff, mesh, and pair style, so tighter settings may be needed for strict reproducibility.

## GROMACS Dataset
The folders "[LysoProtein/](./LysoProtein/)", "[Transmembrane/](./Transmembrane/)", and "[Li-ion-Electrolyte/](./Li-ion-Electrolyte/)" contain the GROMACS input files for the lysozyme protein, the transmembrane bovine bc1 complex, and Li-ion aqueous electrolytes, respectively. The transmembrane input files were downloaded from [MemProt MD](https://memprotmd.bioch.ox.ac.uk/_ref/mpstruc/transmembrane-proteins-alpha-helical/_sim/1sqq_default_dppc/Chain.D/) and have been slightly modified in terms of the .mdp and README files to ensure compatibility with the current version of GROMACS. 

## LAMMPS Dataset
The folder [`LAMMPS-Water/`](./LAMMPS-Water) contains a LAMMPS input script and equilibrated data file for an SPC/E bulk water system. The bundled example script currently reads `equi_bulk.4000000.data`, replicates it by `2 2 2`, and uses `ewald 1e-9` as a high-accuracy reference setup. Commented lines in the script show the corresponding ESP configuration used for comparison.

# Citing

If LAMMPS-ESP is useful in your work, please star this repository and cite both the software and the reference below.

### Preferred citation (BibTeX)
```bibtex
@misc{liang2025arxiv,
  title         = {Accelerating Fast Ewald Summation with Prolates for Molecular Dynamics Simulations},
  author        = {Jiuyang Liang and Libin Lu and Alex Barnett and Leslie Greengard and Shidong Jiang},
  year          = {2025},
  eprint        = {2505.09727},
  archivePrefix = {arXiv},
  primaryClass  = {math.NA},
  url           = {https://arxiv.org/abs/2505.09727}
}
```

# Main developers

- **Jiuyang Liang** — Lead developer  
  *Flatiron Institute, Simons Foundation*
- **Libin Lu**  
  *Flatiron Institute, Simons Foundation*
- **Alex Barnett**  
  *Flatiron Institute, Simons Foundation*
- **Leslie Greengard**  
  *Flatiron Institute, Simons Foundation*
- **Shidong Jiang**  
  *Flatiron Institute, Simons Foundation*

# Related software

GROMACS-ESP is available at https://github.com/lu1and10/Ewald-Splitting-with-Prolates



<!--
<a href="https://info.flagcounter.com/pz9h"><img src="https://s01.flagcounter.com/count2/pz9h/bg_FFFFFF/txt_000000/border_CCCCCC/columns_4/maxflags_12/viewers_0/labels_0/pageviews_0/flags_0/percent_0/" alt="Flag Counter" border="0"></a>
-->
