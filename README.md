# Ewald-Splitting-with-Prolates

"[LAMMPS/](./LAMMPS/)" folder contains a fork of the [LAMMPS](https://github.com/lammps/lammps) molecular dynamics software.

This fork includes custom modifications for the ESP (Ewald summation with prolate spheroidal wave functions) method.

The ESP method code modifications main contributors:
* Jiuyang Liang
* Libin Lu
* Alex Barnett
* Leslie Greengard
* Shidong Jiang

The GROMACS repository is available in [Libin's GitHub profile](https://github.com/lu1and10/Ewald-Splitting-with-Prolates).

## GROMACS Dataset
The folders "[LysoProtein/](./LysoProtein/)", "[Transmembrane/](./Transmembrane/)", and "[Li-ion-Electrolyte/](./Li-ion-Electrolyte/)" contain the GROMACS input files for the lysozyme protein, the transmembrane bovine bc1 complex, and Li-ion aqueous electrolytes, respectively. The transmembrane input files were downloaded from [MemProt MD](https://memprotmd.bioch.ox.ac.uk/_ref/mpstruc/transmembrane-proteins-alpha-helical/_sim/1sqq_default_dppc/Chain.D/) and have been slightly modified in terms of the .mdp and README files to ensure compatibility with the current version of GROMACS. 

## LAMMPS Dataset
The folders "[LAMMPS-Water/](./LAMMPS-Water)" contains LAMMPS input files for the SPC/E bulk water system. The system is replicated 11-fold and 34-fold to generate larger systems containing 3,597,693 and 106,238,712 atoms, respectively.  

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
pair_style lj/cut/coul/long  10 6  # Cutoffs for the LJ and Coulomb interactions
kspace_style pppm 1e-4 # Splitting accuracy
```
To switch to the PSWF, replace them with:
```
pair_style lj/cut/coul/ps  10 6
kspace_style ppps 1e-4 1e-4  # Splitting accuracy and spreading accuracy
```
For specifying the number of Fourier grids and spreading points, you can use:
```
kspace_modify mesh 100 100 100 order 4  # Mesh size along each axis and spreading order
```
Currently, my code supports orders ranging from 2 to 8 and spreading/splitting accuracy between 10^-7 and 10^-1.
## Use ESP method in GROMACS
To be completed ASAP.
# "Optimal" Parameter Sets
To be completed ASAP.

More data files and detailed descriptions will be available shortly. Libin and I are currently working on further HPC improvements. If you have any questions, please do not hesitate to contact me via email at jliang@flatironinstitute.org or llu@flatironinstitute.org.

<a href="https://info.flagcounter.com/pz9h"><img src="https://s01.flagcounter.com/count2/pz9h/bg_FFFFFF/txt_000000/border_CCCCCC/columns_4/maxflags_12/viewers_0/labels_0/pageviews_0/flags_0/percent_0/" alt="Flag Counter" border="0"></a>
