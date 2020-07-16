import glob
import mdtraj as md


for ligand_name in glob.glob('lig*'):
    print(ligand_name)
    for phase in ['solvent']:#, 'complex']:
        for ligand_stat in ['new', 'old']: 
            #traj = md.load(f'{ligand_name}/{ligand_name}.{phase}.{ligand_stat}.forward_snapshots.pdb')
            traj = md.load(f'{ligand_name}/{ligand_name}.{phase}.{ligand_stat}.og_snapshots.pdb')
            print(traj)
            idx = traj.topology.select('not protein and not water')
            traj_solvent = traj.atom_slice(idx)
            for frame_idx, frame in enumerate(traj_solvent):
                frame.save(f'{ligand_name}/{ligand_stat}_{frame_idx}_{phase}.pdb')



