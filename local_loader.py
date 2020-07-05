#!/usr/bin/env python

def local_loader(args):
    """
    wrap all of the local loading functions
    """
    from simtk.openmm.openmm import XmlSerializer
    import pickle
    import numpy as np
    import mdtraj as md
    
    #pull the systems
    with open(args.vacuum_system_filename, 'r') as infile:
        vac_sys = infile.read()
    vacuum_system = XmlSerializer.deserialize(vac_sys)

    with open(args.nonvacuum_system_filename, 'r') as infile:
        nonvac_sys = infile.read()
    nonvac_system = XmlSerializer.deserialize(nonvac_sys)

    #pull the topologies
    with open(args.nonvacuum_topology, 'rb') as f:
        omm_topology = pickle.load(f)
    with open(args.vacuum_topology, 'rb') as f:
        subset_omm_topology = pickle.load(f)

    md_topology = md.Topology.from_openmm(omm_topology)
    subset_md_topology = md.Topology.from_openmm(subset_omm_topology)

    mol_indices = md_topology.select('resname MOL')
    mol_atoms = [atom for atom in md_topology.atoms if atom.index in mol_indices]

    subset_mol_indices = subset_md_topology.select('resname MOL')
    subset_mol_atoms = [atom for atom in subset_md_topology.atoms if atom.index in mol_indices]

    mol_atom_dict = {i:j for i, j in zip(mol_indices, subset_mol_indices)} #mol atom dict is complex-to-vacuum indices

    save_indices = md_topology.select('resname MOL')

    #print what was extracted:
    print(f"atom extractions...")
    print(f"\t(vacuum_atom, atom_in_environment)")
    for subset_atom, atom in zip(subset_mol_atoms, mol_atoms):
        print(f"({subset_atom.name}, {atom.name})")

    return nonvac_system, vacuum_system, mol_atom_dict, md_topology, save_indices
