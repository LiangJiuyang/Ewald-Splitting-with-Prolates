==========
 Contents
==========

em.mdp                  --- MDP file for Energy Minimising the System
pr.mdp                  --- MDP file for Equilibrating the System
100ns.mdp               --- MDP file for 100 ns of Molecular Dynamics
atomistic-system.pdb    --- PDB file of the system
topol.top               --- Topology File for system
atomistic-system.ndx    --- Index file for system
itp/lipids-gmx53a6.itp  --- Gromos53a6 Lipid Parameters from LipidBook
charmm36.ff             --- Charmm36 Forcefield Parameters from http://mackerell.umaryland.edu/charmm_ff.shtml
itp/charmm36-lipids.itp --- Charmm36 Lipid Parameters from http://mackerell.umaryland.edu/charmm_ff.shtml

================
 Example Usage 
================

Energy Minimise:

mkdir EM

gmx grompp -f mdp_files/em.mdp -o EM/em -c atomistic-system.pdb -n atomistic-system.ndx -maxwarn -1

cd EM

gmx mdrun -deffnm em -v

=====================================================================================

Set up Position Restrained Protein MD Simulation (to equilibrate atomistic system):

mkdir PR

gmx grompp -f mdp_files/pr.mdp -o PR/pr -c EM/em.gro -r EM/em.gro -n atomistic-system.ndx -maxwarn -1
cd PR

gmx mdrun -deffnm pr -v

=====================================================================================

Run 100 ns Molecular Dynamics Simulation:

mkdir 100ns

gmx grompp -f mdp_files/100ns.mdp -o 100ns/md -c PR/pr.gro -n atomistic-system.ndx -maxwarn -1

cd 100ns

gmx mdrun -deffnm md -v

=====================================================================================
