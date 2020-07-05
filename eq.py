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

#read args
parser = argparse.ArgumentParser(description='run equilibrium on non-alchemical 0 endstate')
parser.add_argument("system_filename", help='system.xml')
parser.add_argument("positions_box_vectors_filename", help='positions and box vectors filename; is an .npz')
parser.add_argument("initial_index", help='index corresponding to appropriate starting position in positions_filename, box_vectors_filename', type=int)
parser.add_argument("steps_per_application", help='number of MD steps per application', type=int)
parser.add_argument("number_of_applications", help="the number of applications to run and save", type=int)
parser.add_argument("out_npz", help='name of output file to write positions, box_vectors_filename')
args = parser.parse_args()

#pull box vectors/positions
init_arrays = np.load(args.positions_box_vectors_filename)
positions = init_arrays['positions'][args.initial_index,:,:] * unit.nanometers
try:
    box_vectors = init_arrays['box_vectors'][args.initial_index,:,:] * unit.nanometers
except Exception as e:
    print(f"loading box vectors error: {e}")
    box_vectors = None

#pull the system
with open(args.system_filename, 'r') as infile:
    sys_xml = infile.read()
system = XmlSerializer.deserialize(sys_xml)

#create thermostate and sampler_state
from openmmtools.states import ThermodynamicState, SamplerState
pressure = None if box_vectors is None else 1.0*unit.atmosphere
thermostate = ThermodynamicState(system, temperature=300*unit.kelvin, pressure=pressure)
sampler_state = SamplerState(positions) #no velocities yet

#minimize
from perses.dispersed.feptasks import minimize
minimize(thermostate, sampler_state)

#out loggers
out_positions = []
out_box_vectors = []

"""
then we run normal md (without the ani part)
"""
from openmmtools.mcmc import LangevinSplittingDynamicsMove
move = LangevinSplittingDynamicsMove(timestep=2.0 * unit.femtoseconds,
                                     collision_rate = 1.0 / unit.picoseconds,
                                     n_steps = args.steps_per_application,
                                     reassign_velocities = True,
                                     constraint_tolerance=1e-6)
for i in tqdm.trange(args.number_of_applications):
    move.apply(thermostate, sampler_state)
    out_positions.append(sampler_state.positions.value_in_unit_system(unit.md_unit_system))
    if box_vectors is None:
        pass
    else:
        out_box_vectors.append(sampler_state.box_vectors.value_in_unit_system(unit.md_unit_system))

#now we save...
out_positions = np.array(out_positions)
if box_vectors is None:
    np.savez(args.out_npz, positions=out_positions)
else:
    np.savez(args.out_npz, positions=out_positions, box_vectors = np.array(out_box_vectors))
