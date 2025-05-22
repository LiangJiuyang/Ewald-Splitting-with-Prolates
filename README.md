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

The folders "[LysoProtein/](./LysoProtein/)", "[Transmembrane/](./Transmembrane/)", and "[Li-ion-Electrolyte/](./Li-ion-Electrolyte/)" contain the GROMACS input files for the lysozyme protein, the transmembrane bovine bc1 complex, and Li-ion aqueous electrolytes, respectively. The transmembrane input files were downloaded from [MemProt MD](https://memprotmd.bioch.ox.ac.uk/_ref/mpstruc/transmembrane-proteins-alpha-helical/_sim/1sqq_default_dppc/Chain.D/) and have been slightly modified in terms of the .mdp and README files to ensure compatibility with the current version of GROMACS. 

More data files and detailed descriptions will be available shortly. If you have any questions, please do not hesitate to contact me via email at jliang@flatironinstitute.org.

<a href="https://info.flagcounter.com/pz9h"><img src="https://s01.flagcounter.com/count2/pz9h/bg_FFFFFF/txt_000000/border_CCCCCC/columns_4/maxflags_12/viewers_0/labels_0/pageviews_0/flags_0/percent_0/" alt="Flag Counter" border="0"></a>
