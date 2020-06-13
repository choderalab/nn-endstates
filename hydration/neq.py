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
parser = argparse.ArgumentParser(description='run nonequilibrium')
parser.add_argument("direction", help='which direction to run (forward or backward)')
parser.add_argument("vacuum_system_filename", help='vacuum system.xml')
parser.add_argument(f"nonvacuum_system_filename", help='nonvacuum system.xml')
parser.add_argument(f"vacuum_topology", help='vacuum topology.pkl')
parser.add_argument(f"nonvacuum_topology", help='nonvacuum topology.pkl')
parser.add_argument("positions_box_vectors_filename", help='positions and box vectors filename; is an .npz')
parser.add_argument("index_to_run", help='index corresponding to appropriate starting position in positions_filename, box_vectors_filename', type=int)
parser.add_argument("annealing_steps", help='number of MD steps per application', type=int)
parser.add_argument("directory_name", help='directory to save traj and work file')
parser.add_argument(f"traj_work_file_prefix", help='prefix with which to write trajectory and work file')
parser.add_argument(f"write_trajectory_interval", help='interval with which to write trajectory output', type=int)
args = parser.parse_args()

sys.path.insert(1, f"/mnt/c/Users/domin/github/nn-endstates")
from complete import *
from local_loader import local_loader


nonvac_system, vacuum_system, mol_atom_dict, md_topology, save_indices = local_loader(args)

works = annealed_importance_sampling(direction=args.direction,
                                 system = nonvac_system,
                                 system_subset = vacuum_system,
                                 subset_indices_map = mol_atom_dict,
                                 positions_cache_filename = args.positions_box_vectors_filename,
                                 index_to_run = args.index_to_run,
                                 directory_name = args.directory_name,
                                 trajectory_prefix = args.traj_work_file_prefix,
                                 md_topology = md_topology,
                                 steps_per_application = args.annealing_steps,
                                 integrator_kwargs = {'temperature': 300.0 * unit.kelvin,
                                                      'collision_rate': 1.0 / unit.picoseconds,
                                                      'timestep': 1.0 * unit.femtoseconds,
                                                      'splitting': "V R O R F",
                                                      'constraint_tolerance': 1e-6,
                                                      'pressure': 1.0 * unit.atmosphere},
                                 save_indices = save_indices,
                                 position_extractor = None,
                                 write_trajectory_interval=args.write_trajectory_interval)

np.savez(os.path.join(os.getcwd(), args.directory_name, f"{args.traj_work_file_prefix}.works.npz"), works=works)
