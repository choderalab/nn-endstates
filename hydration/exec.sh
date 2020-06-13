#!/usr/bin/bash
pwd

#python eq.py lig01.solvent.system.xml lig01.solvent.positions.npz 0 1000 10 lig01.solvent.eq.positions.npz 

#python neq.py forward lig01.vacuum.system.xml lig01.solvent.system.xml lig01.vacuum.topology.pkl lig01.solvent.topology.pkl lig01.solvent.eq.positions.npz 0 100 lig01_forward lig01.forward.solvent.100_steps 100

#python ANI_eq.py lig01.vacuum.system.xml lig01.solvent.system.xml lig01.vacuum.topology.pkl lig01.solvent.topology.pkl lig01.solvent.positions.npz 0 100 lig01.solvent.eq_ani.positions.npz

python neq.py backward lig01.vacuum.system.xml lig01.solvent.system.xml lig01.vacuum.topology.pkl lig01.solvent.topology.pkl lig01.solvent.eq_ani.positions.npz 0 100 lig01_backward lig01.backward.solvent.100_steps 100


