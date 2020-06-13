# !/usr/bin/env python
import numpy as np
import sys
import pickle
import mdtraj as md
import os
import argparse
from simtk import unit
import tqdm
import mdtraj.utils as mdtrajutils
import mdtraj as md
from simtk.openmm.openmm import XmlSerializer
import tqdm
from pkg_resources import resource_filename


#read args
parser = argparse.ArgumentParser(description='run equilibrium on non-alchemical 1 endstate')
parser.add_argument("vacuum_system_filename", help='vacuum system.xml')
parser.add_argument(f"nonvacuum_system_filename", help='nonvacuum system.xml')
parser.add_argument(f"vacuum_topology", help='vacuum topology.pkl')
parser.add_argument(f"nonvacuum_topology", help='nonvacuum topology.pkl')
parser.add_argument("positions_box_vectors_filename", help='positions and box vectors filename; is an .npz')
parser.add_argument("index_to_run", help='index corresponding to appropriate starting position in positions_filename, box_vectors_filename', type=int)
parser.add_argument("number_steps", help='number of MD steps', type=int)
parser.add_argument("out_npz", help='name of output file to write positions, box_vectors_filename')
args = parser.parse_args()

sys.path.insert(1, f"/mnt/c/Users/domin/github/nn-endstates")
from complete import *
from local_loader import local_loader

nonvac_system, vacuum_system, mol_atom_dict, md_topology, save_indices = local_loader(args)

particle_state, reduced_energy_array =  ANI_endstate_sampler(
                                                                system = nonvac_system,
                                                                system_subset = vacuum_system,
                                                                subset_indices_map = mol_atom_dict,
                                                                positions_cache_filename = args.positions_box_vectors_filename,
                                                                md_topology = md_topology,
                                                                index_to_run = args.index_to_run,
                                                                steps_per_application = args.number_steps,
                                                                integrator_kwargs = {'temperature': 300.0 * unit.kelvin,
                                                                                      'collision_rate': 1.0 / unit.picoseconds,
                                                                                      'timestep': 2.0 * unit.femtoseconds,
                                                                                      'splitting': "V R O R F",
                                                                                      'constraint_tolerance': 1e-6,
                                                                                      'pressure': 1.0 * unit.atmosphere}
                        )

out_positions = []
out_box_vectors = []

out_positions.append(particle_state.positions.value_in_unit_system(unit.md_unit_system))
if particle_state.box_vectors is None:
    pass
else:
    out_box_vectors.append(particle_state.box_vectors.value_in_unit_system(unit.md_unit_system))

#now we save...
out_positions = np.array(out_positions)
if particle_state.box_vectors is None:
    np.savez(args.out_npz, positions=out_positions, energies = reduced_energy_array)
else:
    np.savez(args.out_npz, positions=out_positions, box_vectors = np.array(out_box_vectors), energies = reduced_energy_array)
