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
GROMACS ($>85,000$ citations), and NAMD
all rely on Ewald-based approaches, such as the (Smooth) Particle-Mesh Ewald ([PME/SPME](https://scholar.google.com/citations?user=y4Ts-LkAAAAJ&hl=en), $>57,000$ citations)
and the Particle-Particle-Particle-Mesh (PPPM, $>12000$ citations) methods. 

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

## Use ESP method in LAMMPS
The compilation process follows the standard LAMMPS build commands. Below is an example:
```
mkdir LAMMPS/build
cd LAMMPS/build
cmake -C ../cmake/presets/oneapi.cmake -D PKG_RIGID=on -D PKG_MOLECULE=on -D PKG_KSPACE=on ../cmake
make install
make -j 4 
```
To use the PPPM, include the following commands in the input file:
```
pair_style lj/cut/coul/long  9 9  # Cutoffs for the LJ and Coulomb interactions
kspace_style pppm 1e-4 # Splitting accuracy
```
To switch to the PSWF, replace them with:
```
pair_style lj/cut/coul/ps  9 9
kspace_style ppps 1e-4 1e-4  # Splitting accuracy and spreading accuracy
```
For specifying the number of Fourier grids and spreading points, you can use:
```
kspace_modify mesh 100 100 100 order 4  # Mesh size along each axis and spreading order
```

## GROMACS Dataset
The folders "[LysoProtein/](./LysoProtein/)", "[Transmembrane/](./Transmembrane/)", and "[Li-ion-Electrolyte/](./Li-ion-Electrolyte/)" contain the GROMACS input files for the lysozyme protein, the transmembrane bovine bc1 complex, and Li-ion aqueous electrolytes, respectively. The transmembrane input files were downloaded from [MemProt MD](https://memprotmd.bioch.ox.ac.uk/_ref/mpstruc/transmembrane-proteins-alpha-helical/_sim/1sqq_default_dppc/Chain.D/) and have been slightly modified in terms of the .mdp and README files to ensure compatibility with the current version of GROMACS. 

## LAMMPS Dataset
The folders "[LAMMPS-Water/](./LAMMPS-Water)" contains LAMMPS input files for the SPC/E bulk water system. The system is replicated 11-fold and 34-fold to generate larger systems containing 3,597,693 and 106,238,712 atoms, respectively.  

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
