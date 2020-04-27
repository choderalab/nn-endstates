# !/usr/bin/env python

from complete import *
import sys
from pkg_resources import resource_filename
import pickle
import mdtraj as md
import os
"""

execute:

Arguments:
    1:factory_np_filename: str
        '.npy' name of factory dict file
    2:phase: str
        phase of the simulation (key of the factory dict); can be 'solvent' or 'complex'
    3:endstate: str
        'new' or 'old'
    4:steps_per_application: int
        number of annealing steps to run; instantaneous is 1
    5:number_of_applications: int
        number of annealing protocols to run
    6:endstate_cache_filename: str
        path to mdtraj loadable trajectory from which endstates will be extracted
    7:write_dir_name : str
        directory name of trajectory cache
    8:trajectory_prefix : str
        prefix name of trajectory files written to write_dir_name
    9:write_trajectory_interval : int, default 1
        how often to write trajectory to disk
"""


#pull the command line arguments
factory_np_filename = sys.argv[1] 
phase = sys.argv[2]
endstate = sys.argv[3]
steps_per_application = sys.argv[4]
number_of_applications = sys.argv[5]
endstate_cache_filename = sys.argv[6]
write_dir_name = sys.argv[7]
trajectory_prefix = sys.argv[8]

try:
    write_trajectory_interval = int(sys.argv[9])
except Exception as e:
    write_trajectory_interval=1

#extract the factory dictionary from a .npy file
print(f"loading factory dict...")
factory_dict = np.load(factory_np_filename, allow_pickle=True).item()
factory = factory_dict[phase]
subset_factory = factory_dict['vacuum']

#extract system and vacuum system
print(f"defining systems")
system = getattr(factory._topology_proposal, f"_{endstate}_system")
system_subset = getattr(subset_factory._topology_proposal, f"_{endstate}_system")

#now we have to extract the appropriate subset indices...
print(f"extracting appropriate subset indices...")
omm_topology = getattr(factory._topology_proposal, f"_{endstate}_topology")
subset_omm_topology = getattr(subset_factory._topology_proposal, f"_{endstate}_topology")

md_topology = md.Topology.from_openmm(omm_topology)
subset_md_topology = md.Topology.from_openmm(subset_omm_topology)

mol_indices = md_topology.select('resname MOL')
mol_atoms = [atom for atom in md_topology.atoms if atom.index in mol_indices]

subset_mol_indices = subset_md_topology.select('resname MOL')
subset_mol_atoms = [atom for atom in subset_md_topology.atoms if atom.index in mol_indices]

mol_atom_dict = {i:j for i, j in zip(mol_indices, subset_mol_indices)} #mol atom dict is complex-to-vacuum indices

#print what was extracted:
print(f"atom extractions...")
print(f"\t(vacuum_atom, atom_in_environment)")
for subset_atom, atom in zip(subset_mol_atoms, mol_atoms):
    print(f"({subset_atom.name}, {atom.name})")


works = annealed_importance_sampling(system = system,
                                 system_subset = system_subset,
                                 subset_indices_map = mol_atom_dict,
                                 endstate_cache_filename = endstate_cache_filename,
                                 directory_name = write_dir_name,
                                 trajectory_prefix = trajectory_prefix,
                                 md_topology = md_topology,
                                 number_of_applications = int(number_of_applications),
                                 steps_per_application = int(steps_per_application),
                                 integrator_kwargs = {'temperature': 300.0 * unit.kelvin,
                                                      'collision_rate': 1.0 / unit.picoseconds,
                                                      'timestep': 1.0 * unit.femtoseconds,
                                                      'splitting': "V R O R F",
                                                      'constraint_tolerance': 1e-6,
                                                      'pressure': 1.0 * unit.atmosphere},
                                 save_indices = None,
                                 position_extractor = None #solvent_factory.new_positions
                                 write_trajectory_write_trajectory_interval
                                )

save_to = os.path.join(os.getcwd(), write_dir_name, f"{trajectory_prefix}.works.npy")
np.save(save_to, np.array(works))
print(f"successfully saved work array to {save_to}; terminating")

