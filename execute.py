#!/usr/bin/env python
# coding: utf-8

# ## Annealed Importance Sampling to exchange intramolecular talks between torchANI and openmm.

import os

import mdtraj as md
import torch
import torchani
from openmmtools.constants import kB
from simtk import unit

atomic_num_to_symbol_dict = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
mass_dict_in_daltons = {'H': 1.0, 'C': 12.0, 'N': 14.0, 'O': 16.0}
import numpy as np
import mdtraj.utils as mdtrajutils
from openmmtools.states import ThermodynamicState, SamplerState

from perses.dispersed.utils import configure_platform
from openmmtools import cache, utils

cache.global_context_cache.platform = configure_platform(utils.get_fastest_platform().getName())

import logging

logging.basicConfig(level=logging.NOTSET)
_logger = logging.getLogger("interpolANI")
_logger.setLevel(logging.DEBUG)


class ANI1_force_and_energy(object):
    # some class attributes
    mass_unit = unit.dalton
    distance_unit = unit.nanometers
    time_unit = unit.femtoseconds
    energy_unit = unit.kilojoules_per_mole
    ani_distance_unit = unit.angstroms
    hartree_to_kJ_per_mole = 2625.499638
    ani_energy_unit = hartree_to_kJ_per_mole * energy_unit
    nm_to_angstroms = 10.
    angstroms_to_nm = 1e-1

    def __init__(self,
                 model,
                 atoms,
                 platform='cpu',
                 temperature=300 * unit.kelvin
                 ):

        """
        Performs energy and force calculations.
        Slightly modified code from: 
            https://gist.github.com/wiederm/7ac5c29e5a0dea9d17ef16dda93fe02d#file-reweighting-py-L42; thanks, Marcus
        
        Parameters
        ----------
        model: torchani.models object
            model from which to compute energies and forces
        atoms: str
            a string of atoms in the indexed order
        platform : str, default 'cpu',
            platform on which to initialize the model device
        temperature : float * unit.kelvin, default 300 * unit.kelvin
            temperature
        """
        self.model = model
        self.atoms = atoms

        self.platform = platform
        self.device = torch.device(self.platform)
        if self.platform == 'cpu':
            torch.set_num_threads(2)
        else:
            raise Exception(f"we don't support gpu just yet")

        self.species = self.model.species_to_tensor(atoms).to(self.device).unsqueeze(0)
        self.temperature = temperature
        self.beta = 1.0 / (kB * temperature)

        self.W_shads = []
        self.W = []

    def minimize(self,
                 coords: unit.quantity.Quantity,
                 maxiter: int = 1000):
        """
        Minimizes the molecule. Note, we usually don't want to do this given an input structure since they are already distributed i.i.d.
        
        Parameters
        ----------
        coords:simtk.unit.quantity.Quantity
        maxiter: int
            Maximum number of minimization steps performed.

        Returns
        -------
        coords:simtk.unit.quantity.Quantity
        """
        from scipy import optimize
        assert (type(coords) == unit.quantity.Quantity)

        x = coords.value_in_unit(unit.angstrom)
        self.memory_of_energy = []
        print("Begin minimizing...")
        f = optimize.minimize(self._target_energy_function, x, method='BFGS',
                              jac=True, options={'maxiter': maxiter, 'disp': True})

        _logger.critical(f"Minimization status: {f.success}")
        memory_of_energy = copy.deepcopy(self.memory_of_energy)
        self.memory_of_energy = []

        return f.x.reshape(-1, 3) * unit.angstrom, memory_of_energy

    def calculate_force(self,
                        x: unit.quantity.Quantity) -> (unit.quantity.Quantity, unit.quantity.Quantity):
        """
        Given a coordinate set the forces with respect to the coordinates are calculated.
        
        Parameters
        ----------
        x : array of floats, unit'd (distance unit)
            initial configuration
            
        Returns
        -------
        F : float, unit'd
        E : float, unit'd
        """
        assert (type(x) == unit.quantity.Quantity)

        coordinates = torch.tensor([x.value_in_unit(unit.angstroms)],
                                   requires_grad=True, device=self.device, dtype=torch.float32)

        energy_in_hartree = self._calculate_energy(coordinates)

        # derivative of E (in kJ/mol) w.r.t. coordinates (in nm)
        derivative = torch.autograd.grad((energy_in_hartree).sum(), coordinates)[0]

        if self.platform == 'cpu':
            F = -1 * derivative[0].numpy()
        elif self.platform == 'cuda':
            F = - np.array(derivative.cpu())[0]
        else:
            raise RuntimeError('Platform needs to be specified. Either CPU or CUDA.')

        return (F * self.hartree_to_kJ_per_mole * self.nm_to_angstroms * (unit.kilojoule_per_mole / unit.nanometer),
                energy_in_hartree.item() * self.hartree_to_kJ_per_mole * unit.kilojoule_per_mole)

    def _calculate_energy(self, coordinates: torch.tensor):
        """
        Helpter function to return energies as tensor.
        Given a coordinate set the energy is calculated.
        
        Parameters
        ----------
        coordinates : torch.tensor 
            coordinates in angstroms without units attached
            
        Returns
        -------
        energy_in_hartree : torch.tensor
            
        """

        # stddev_in_hartree = torch.tensor(0.0,device = self.device, dtype=torch.float64)
        energy_in_hartree = self.model((self.species, coordinates)).energies

        return energy_in_hartree

    def _target_energy_function(self, x) -> float:
        """
        Given a coordinate set (x) the energy is calculated in kJ/mol.
        
        Parameters
        ----------
        x : array of floats, unit'd (distance unit)
            initial configuration
            
        Returns
        -------
        E : float, unit'd 
        """
        x = x.reshape(-1, 3) * unit.angstrom
        F, E = self.calculate_force(x)
        F_flat = -np.array(F.value_in_unit(unit.kilojoule_per_mole / unit.angstrom).flatten(), dtype=np.float64)
        self.memory_of_energy.append(E)
        return E.value_in_unit(unit.kilojoule_per_mole), F_flat

    def calculate_energy(self, x: unit.Quantity):
        """
        Given a coordinate set (x) the energy is calculated in kJ/mol.
        Parameters
        ----------
        x : array of floats, unit'd (angstroms)
            initial configuration
            
        Returns
        -------
        energy : unit.quantity.Quantity
            energy in kJ/mol
        """

        assert (type(x) == unit.quantity.Quantity)
        coordinates = torch.tensor([x.value_in_unit(unit.angstroms)],
                                   requires_grad=True, device=self.device, dtype=torch.float32)

        energy_in_hartrees = self._calculate_energy(coordinates)
        energy = energy_in_hartrees.item() * self.hartree_to_kJ_per_mole * unit.kilojoule_per_mole
        return energy


class InterpolAIS(object):
    """
    Class to build hybrid OpenMM/TorchANI system and conduct Annealed Importance Sampling (AIS) between the full-openMM system and the hybrid OpenMM/TorchAni system that respects the following rules:
        1.  all intermolecular talks happen with sterics
        2.  all intramolecular talks of the `ligand` of interest happen with OpenMM at lambda = 0 and with torchANI at lambda = 1
        3.  all environment atoms talk to each other with OpenMM for all lambdas
        
    Example:
    
        input : complex_system, complex_topology
        output : trajectory, particle_works
    
        
    
    """

    def __init__(self,
                 complex_system,
                 complex_topology,
                 ligand_system,  # note this must be unique
                 ligand_subset_indices,
                 temperature=300 * unit.kelvin,
                 pressure=1 * unit.atmosphere,
                 trajectory_directory='test',
                 trajectory_prefix='trial_1',
                 atom_selection_string='not water',
                 lambda_sequence=np.array([0, 1]),
                 ani_model=torchani.models.ANI1ccx(),
                 platform='cpu'):
        """
        create vacuum_openmm_ligand system, vacuum_torchani_ligand system, and logger object
        
        Arguments:
            complex_system : openmm.System 
                openmm system of the complex
            complex_topology : openmm.Topology
                topology of the complex
            ligand_system : openmm.System
                parametrized ligand in vacuum
            ligand_subset_indices : list of int
                complex topology ligand atom indices
            temperature : unit.quantity.Quantity, (units compatible with unit.kelvin)
                temperature
            pressure : unit.quantity.Quantity, (units compatible with unit.atmosphere)
                pressure
            trajectory_directory : str
                path of written trajectory
            trajectory_prefix : str
                name of traj prefix
            atom_selection_string : str, default 'not water'
                atom indices to save (mdtraj str)
            lambda_sequence : np.array, default np.array([0,1])
                sequence of lamda values
            ani_model: torchani.models object
                model from which to compute energies and forces
            platform : str, default 'cpu'
                which platform to use for ani
            
        """
        import numpy as np
        import os
        import simtk.openmm as openmm

        self.context_cache = cache.global_context_cache

        self.lambda_sequence = lambda_sequence

        self.complex_system = complex_system
        self.complex_topology = complex_topology

        self.ligand_system = ligand_system
        assert ligand_subset_indices == range(min(ligand_subset_indices), max(
            ligand_subset_indices) + 1), f"ligand subset indices ({ligand_subset_indices}) is not a sequence"
        self.ligand_subset_indices = ligand_subset_indices

        # pull masses
        _logger.debug(f"defining a mass matrix")
        assert len(
            ligand_subset_indices) == self.ligand_system.getNumParticles(), f"the number of ligand subset indices is _not_ equal to the number of ligand system particles ({self.ligand_system.getNumParticles()})"
        for index, lig_subset_index in enumerate(ligand_subset_indices):
            assert self.complex_system.getParticleMass(lig_subset_index) == self.ligand_system.getParticleMass(
                index), f"masses do not match"
        self.mass_matrix = np.array([[self.complex_system.getParticleMass(q).value_in_unit(unit.daltons) for q in
                                      range(self.complex_system.getNumParticles())]]) * unit.daltons

        # create a thermodynamic and sampler state
        _logger.debug(f"defining thermostates")
        self.temperature = temperature
        self.pressure = pressure
        self.complex_thermostate = ThermodynamicState(system=self.complex_system, temperature=temperature,
                                                      pressure=pressure)
        self.ligand_thermostate = ThermodynamicState(system=self.ligand_system, temperature=temperature)

        #         self.complex_sampler_state = SamplerState(positions = complex_positions, box_vectors = self.complex_system.getDefaultPeriodicBoxVectors())
        #         ligand_positions = unit.Quantity(np.zeros([len(self.ligand_subset_indices), 3]), unit=unit.nanometer)
        #         for idx in ligand_subset:
        #             ligand_positions[idx, :] = self.complex_sampler_state.positions[idx,:]

        #         self.ligand_sampler_state = SamplerState(positions = ligand_positions, box_vectors = self.ligand_system.getDefaultPeriodicBoxVectors())

        # trajectory maker
        self.trajectory_directory, self.trajectory_prefix = trajectory_directory, trajectory_prefix
        if self.trajectory_directory is not None and self.trajectory_prefix is not None:
            _logger.debug(f"creating trajectory storage object")
            self.write_traj = True
            self.neq_traj_filename = os.path.join(os.getcwd(), self.trajectory_directory,
                                                  f"{self.trajectory_prefix}.neq")
            os.mkdir(os.path.join(os.getcwd(), self.trajectory_directory))
            complex_md_topology = md.Topology().from_openmm(self.complex_topology)
            ligand_md_topology = complex_md_topology.subset(self.ligand_subset_indices)
            atom_selection_indices = complex_md_topology.select(atom_selection_string)
            self.atom_selection_indices = atom_selection_indices
            self.complex_md_topology = complex_md_topology.subset(self.atom_selection_indices)
            self.ligand_md_topology = ligand_md_topology

        else:
            _logger.debug(f"omitting trajectory storage object")
            self.write_traj = False
            self.neq_traj_filename = None
            self.atom_selection_indices = None
            self.complex_md_topology, self.ligand_md_topology = None, None

        # make the ani model
        _logger.debug(f"creating ANI handler")
        species_str = ''.join(
            [atom.element.symbol for atom in self.complex_topology.atoms() if atom.index in self.ligand_subset_indices])
        self.ani_handler = ANI1_force_and_energy(model=ani_model,
                                                 atoms=species_str,
                                                 platform='cpu',
                                                 temperature=self.temperature)

        # make the openmm model
        self.integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)

        # recorders
        _logger.debug(f"handling recorder objects")
        self.incremental_works = []
        self.shadow_works = []
        self.anneal_num = 0

    def _compute_hybrid_potential(self,
                                  lambda_index):
        """
        function to compute the hybrid reduced potential defined as follows:
        U(x_rec, x_lig) = u_mm,rec(x_rec) - lambda*u_mm,lig(x_lig) + lambda*u_ani,lig(x_lig)
        """
        ligand_state = self.ligand_context.getState(getPositions=True)
        ligand_coords = ligand_state.getPositions(asNumpy=True)
        reduced_potential = (self.complex_thermostate.reduced_potential(self.complex_context)
                             - self.lambda_sequence[lambda_index] * self.ligand_thermostate.reduced_potential(
                    self.ligand_context)
                             + self.lambda_sequence[lambda_index] * self.ani_handler.calculate_energy(
                    ligand_coords) * self.complex_thermostate.beta)
        return reduced_potential

    def minimize(self, complex_positions):
        from perses.dispersed.utils import minimize
        complex_sampler_state, ligand_sampler_state = self._create_sampler_states(complex_positions)
        minimize(self.complex_thermostate, complex_sampler_state)
        return complex_sampler_state.positions

    def _compute_hybrid_forces(self,
                               lambda_index):
        """
        function to compute a hybrid force matrix of shape num_particles x 3
        in the spirit of the _compute_hybrid_potential, we compute the forces in the following way
            F(x_rec, x_lig) = F_mm(x_rec, x_lig) - lambda * F_mm(x_lig) + lambda * F_ani(x_lig)
        """
        # get the complex mm forces
        complex_state = self.complex_context.getState(getForces=True)
        complex_mm_force_matrix = complex_state.getForces(asNumpy=True)

        # get the ligand mm forces
        ligand_state = self.ligand_context.getState(getForces=True)
        ligand_mm_force_matrix = ligand_state.getForces(asNumpy=True)

        # get the ligand ani forces
        ligand_state = self.ligand_context.getState(getPositions=True)
        coords = ligand_state.getPositions(asNumpy=True)
        ligand_ani_force_matrix, energie = self.ani_handler.calculate_force(coords)

        # now combine the ligand forces
        ligand_force_matrix = self.lambda_sequence[lambda_index] * (ligand_ani_force_matrix - ligand_mm_force_matrix)

        # and append to the complex forces...
        complex_mm_force_matrix[self.ligand_subset_indices, :] += ligand_force_matrix

        complex_force_matrix = complex_mm_force_matrix
        return complex_force_matrix

    def _create_sampler_states(self, complex_positions):
        """
        simple utility function to generate complex and ligand sampler states
        """
        from openmmtools.states import SamplerState
        complex_sampler_state = SamplerState(positions=complex_positions,
                                             box_vectors=self.complex_system.getDefaultPeriodicBoxVectors())
        ligand_positions = unit.Quantity(np.zeros([len(self.ligand_subset_indices), 3]), unit=unit.nanometer)
        ligand_positions[self.ligand_subset_indices, :] = complex_sampler_state.positions[self.ligand_subset_indices, :]
        ligand_sampler_state = SamplerState(positions=ligand_positions,
                                            box_vectors=self.ligand_system.getDefaultPeriodicBoxVectors())

        return complex_sampler_state, ligand_sampler_state

    def simulate_baoab(self,
                       _lambda_index,
                       n_steps=1,
                       gamma=1. / unit.picoseconds,
                       dt=1.0 * unit.femtoseconds,
                       metropolize=False):
        """
        Simulate n_steps of BAOAB, accumulating heat
        
        args
            n_steps : int, default 1
                number of steps of dynamics to run
            gamma : unit.quantity.Quantity (units compatible with 1.0/unit.picoseconds)
                collision rate
            dt : unit.quantity.Quantity (units compatible with units.femtoseconds)
                timestep
        """
        Q = 0
        W_shads = 0.
        _complex_sampler_state, _ligand_sampler_state = SamplerState.from_context(
            self.complex_context), SamplerState.from_context(self.ligand_context)
        x_unit, v_unit, t_unit, m_unit = unit.nanometers, unit.nanometers / unit.femtoseconds, unit.femtoseconds, unit.daltons
        x, v = _complex_sampler_state.positions.value_in_unit(
            unit.nanometers), _complex_sampler_state.velocities.value_in_unit(unit.nanometers / unit.femtoseconds)
        dt = dt.value_in_unit(t_unit)
        gamma = gamma * t_unit
        mass_matrix = self.mass_matrix.value_in_unit(m_unit)

        def compute_kinetic_energy(v):
            return 0.5 * np.sum(
                np.matmul(np.square(v.T), mass_matrix.T)) * m_unit * v_unit ** 2 * self.complex_thermostate.beta

        def sample_maxwell_boltzmann():
            return np.sqrt((1. / self.complex_thermostate.beta).value_in_unit(m_unit * v_unit ** 2)) * (
                        np.random.randn(mass_matrix.shape[1], 3) / np.sqrt(mass_matrix)[0, :][:, None])

        def V(v):
            force = self._compute_hybrid_forces(_lambda_index).value_in_unit(m_unit * x_unit / (t_unit ** 2))
            dv = (dt / 2.) * (force / mass_matrix[0, :][:, None])
            return v + dv

        def R(x, v):
            dx = (dt / 2.0) * v
            return x + dx

        def OU(v):
            mb_sample = sample_maxwell_boltzmann()
            _logger.debug(f"\t\t\t\tmaxwell boltzmann sample KE: {compute_kinetic_energy(mb_sample)}")
            return a * v + b * mb_sample

        if metropolize:
            v = sample_maxwell_boltzmann()
        old_potential, old_ke = self._compute_hybrid_potential(_lambda_index), compute_kinetic_energy(v)
        # 0.5 * np.sum(np.matmul(mass_matrix, np.square(v))) * m_unit * v_unit**2 * self.complex_thermostate.beta
        E_old = old_potential + old_ke
        x_old, v_old = x, v
        _logger.debug(f"\t\t\tbefore propagation u, ke, E_tot: {old_potential}, {old_ke},  {old_potential + old_ke}")

        # Mixing parameters for half O step
        a = np.exp(-gamma * dt)
        b = np.sqrt(1 - np.exp(-2 * gamma * dt))
        _logger.debug(f"\t\t\ta, b scales: {a}, {b}")

        _logger.debug(f"\t\t\tbeginning integration...")
        for i in range(n_steps):
            #####################################
            #####################################

            # V step
            v = V(v)
            _logger.debug(f"\t\t\t\tV")
            _logger.debug(f"\t\t\t\t\tupdated ke: {compute_kinetic_energy(v)}")

            # R step
            x = R(x, v)
            _logger.debug(f"\t\t\t\tR")

            # O step
            _logger.debug(f"\t\t\t\tO")
            ke_old = compute_kinetic_energy(v)
            _logger.debug(f"\t\t\t\tke_old: {ke_old}")

            v = OU(v)

            ke_new = compute_kinetic_energy(v)
            _logger.debug(f"\t\t\t\tke_new: {ke_new}")

            Q += (ke_new - ke_old)
            _logger.debug(f"\t\t\t\t heat change: {Q}")

            # R step
            x = R(x, v)
            _logger.debug(f"\t\t\t\tR")

            # update contexts before computing forces again
            _complex_sampler_state.positions = x * x_unit
            _complex_sampler_state.velocities = v * v_unit
            _complex_sampler_state.apply_to_context(self.complex_context)
            _ligand_sampler_state.positions = x[self.ligand_subset_indices, :] * x_unit
            _ligand_sampler_state.apply_to_context(self.ligand_context)

            # V step
            v = V(v)
            _logger.debug(f"\t\t\t\tV")
            _logger.debug(f"\t\t\t\t\tupdated ke: {compute_kinetic_energy(v)}")

            # Update W_shads
            new_potential, new_ke = self._compute_hybrid_potential(_lambda_index), compute_kinetic_energy(v)
            E_new = new_potential + new_ke
            W_shads += (E_new - E_old - Q)

            #####################################
            #####################################

        # update_contexts
        _logger.debug(f"\t\t\tafter propagation u, ke, E_new: {new_potential}, {new_ke}, {new_potential + new_ke}")
        acceptance_probability = min(1., np.exp(E_old - E_new))
        _logger.debug(f"\t\t\tacceptance_probability: {acceptance_probability}")
        if (metropolize and acceptance_probability >= np.random.rand()) or (not metropolize):
            _logger.debug(f"\t\t\taccepted")
            _complex_sampler_state.velocities = v * v_unit
            _complex_sampler_state.apply_to_context(self.complex_context)
        else:
            _logger.debug(f"\t\t\trejected")
            _complex_sampler_state.positions = x_old * x_unit
            _complex_sampler_state.velocities = v_old * v_unit
            _ligand_sampler_state.positions = x_old[self.ligand_subset_indices, :] * x_unit
            _complex_sampler_state.apply_to_context(self.complex_context)
            _ligand_sampler_state.apply_to_context(self.ligand_context)
            Q, W_shads = 0., 0.
        _logger.debug(f"\t\tshadow_work: {W_shads}")

        return Q, W_shads

    def _record_trajectory(self, complex_sampler_state):
        """
        wrapper to record sampler state
        """
        if self.write_traj:
            self.trajectory_positions.append(
                complex_sampler_state.positions[self.atom_selection_indices, :].value_in_unit_system(
                    unit.md_unit_system))
            a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(
                *complex_sampler_state.box_vectors)
            self.box_lengths.append([a, b, c])
            self.box_angles.append([alpha, beta, gamma])

    def anneal(self,
               complex_positions,
               n_steps_per_propagation=1,
               gamma=1. / unit.picoseconds,
               dt=0.5 * unit.femtoseconds,
               metropolize_propagator=False):

        _logger.debug(f"conducting annealing...")
        import copy
        # first make contexts and sampler states
        _logger.debug(f"creating contexts and sampler states...")
        complex_sampler_state, ligand_sampler_state = self._create_sampler_states(complex_positions)
        self.complex_context, complex_integrator = self.context_cache.get_context(self.complex_thermostate,
                                                                                  copy.deepcopy(self.integrator))
        self.ligand_context, ligand_integrator = self.context_cache.get_context(self.ligand_thermostate,
                                                                                copy.deepcopy(self.integrator))
        complex_sampler_state.apply_to_context(self.complex_context)
        self.complex_context.setVelocitiesToTemperature(self.complex_thermostate.temperature)
        complex_sampler_state.update_from_context(self.complex_context)

        ligand_sampler_state.apply_to_context(self.ligand_context)

        # then we have to make some recording objects
        incremental_works = [0.]
        shadow_works = [0.]
        self.trajectory_positions = []
        self.box_lengths, self.box_angles = [], []
        self._record_trajectory(complex_sampler_state)

        reduced_potential = self._compute_hybrid_potential(
            lambda_index=0)
        _logger.debug(f"initial hybrid reduced potential: {reduced_potential}")

        _logger.debug(f"conducting annealing protocol for lambda sequence...")
        for _lambda_index in range(1, len(self.lambda_sequence)):
            _logger.debug(f"\tlambda value: {self.lambda_sequence[_lambda_index]}")

            # first thing to do is update the reduced potential
            new_reduced_potential = self._compute_hybrid_potential(
                lambda_index=_lambda_index)
            _logger.debug(f"\t\tnew reduced potential: {new_reduced_potential}")

            incremental_work = new_reduced_potential - reduced_potential
            _logger.debug(f"\t\tincremental_work: {incremental_work}")
            incremental_works.append(new_reduced_potential - reduced_potential)

            # now we can propagate
            _logger.debug(f"\t\tcomplex_positions_sample: {complex_sampler_state.positions[0, :]}")
            _logger.debug(f"\t\tcomplex_velocities_sample: {complex_sampler_state.velocities[0, :]}")
            _logger.debug(f"\t\tpropagating particle {n_steps_per_propagation} steps...")
            try:
                Q, schatten = self.simulate_baoab(
                    _lambda_index,
                    n_steps_per_propagation,
                    gamma,
                    dt,
                    metropolize_propagator)

                shadow_works.append(schatten)
                ligand_sampler_state.update_from_context(self.ligand_context)
                complex_sampler_state.update_from_context(self.complex_context)
                _logger.debug(f"\t\tcomplex_positions_sample updated: {complex_sampler_state.positions[0, :]}")
                _logger.debug(f"\t\tcomplex_velocities_sample updated: {complex_sampler_state.velocities[0, :]}")

                self._record_trajectory(complex_sampler_state)

                reduced_potential = self._compute_hybrid_potential(
                    lambda_index=_lambda_index)

            except Exception as e:
                _logger.warning(f"{e}")
                break

        # now that we are out of the loop, we can write to the recorders
        self.incremental_works.append(incremental_works)
        self.shadow_works.append(shadow_works)
        if self.write_traj:
            try:
                trajectory = md.Trajectory(np.array(self.trajectory_positions), self.complex_md_topology,
                                           unitcell_lengths=np.array(self.box_lengths),
                                           unitcell_angles=np.array(self.box_angles))
                trajectory.center_coordinates()
                trajectory.save(f"{self.neq_traj_filename}.{self.anneal_num}.pdb")
            except Exception as e:
                _logger.warning(f"{e}")
                _shape = np.array(self.trajectory_positions).shape
                self.trajectory_positions = np.array(self.trajectory_positions)
                to_val = int(np.floor(0.9 * _shape[0]))
                trajectory = md.Trajectory(np.array(self.trajectory_positions[:to_val, :]), self.complex_md_topology,
                                           unitcell_lengths=np.array(self.box_lengths)[:to_val, :],
                                           unitcell_angles=np.array(self.box_angles)[:to_val, :])
                trajectory.center_coordinates()
                trajectory.save(f"{self.neq_traj_filename}.{self.anneal_num}.pdb")
        _logger.debug(f"incremental works: {incremental_works}")

        self.anneal_num += 1


def build_solvated_model(smiles):
    from perses.utils.openeye import smiles_to_oemol
    from openmoltools.forcefield_generators import generateTopologyFromOEMol
    from openmmforcefields.generators import SystemGenerator
    import simtk.openmm.app as app
    from openforcefield.topology import Molecule
    import simtk.openmm as openmm

    forcefield_files = ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml']
    oemol = smiles_to_oemol(smi)
    molecules = [Molecule.from_openeye(oemol)]

    coords = oemol.GetCoords()
    vac_pos = np.array([list(coords[i]) for i, j in coords.items()]) * unit.angstroms
    vac_top = generateTopologyFromOEMol(oemol)

    vac_barostat = None
    vac_system_generator = SystemGenerator(forcefield_files,
                                           barostat=vac_barostat,
                                           forcefield_kwargs={'removeCMMotion': False,
                                                              'ewaldErrorTolerance': 1e-4,
                                                              'nonbondedMethod': app.NoCutoff,
                                                              'constraints': None,
                                                              'hydrogenMass': 4 * unit.amus},
                                           small_molecule_forcefield='gaff-2.11',
                                           molecules=molecules, cache=None
                                           )

    solv_barostat = openmm.MonteCarloBarostat(1.0 * unit.atmosphere, 300 * unit.kelvin, 50)
    solv_system_generator = SystemGenerator(forcefield_files,
                                            barostat=solv_barostat,
                                            forcefield_kwargs={'removeCMMotion': False,
                                                               'ewaldErrorTolerance': 1e-4,
                                                               'nonbondedMethod': app.PME,
                                                               'constraints': None,
                                                               'rigidWater': False,
                                                               'hydrogenMass': 4 * unit.amus},
                                            small_molecule_forcefield='gaff-2.11',
                                            molecules=molecules, cache=None
                                            )

    modeller = app.Modeller(vac_top, vac_pos)
    modeller.addSolvent(solv_system_generator.forcefield, model='tip3p', padding=9 * unit.angstroms,
                        ionicStrength=0.15 * unit.molar)
    solvated_topology = modeller.getTopology()
    solvated_positions = modeller.getPositions()

    # canonicalize the solvated positions: turn tuples into np.array
    solvated_positions = unit.quantity.Quantity(
        value=np.array([list(atom_pos) for atom_pos in solvated_positions.value_in_unit_system(unit.md_unit_system)]),
        unit=unit.nanometers)
    solvated_topology = solvated_topology
    solvated_system = solv_system_generator.create_system(solvated_topology)

    vacuum_positions = unit.quantity.Quantity(
        value=np.array([list(atom_pos) for atom_pos in vac_pos.value_in_unit_system(unit.md_unit_system)]),
        unit=unit.nanometers)
    vacuum_topology = vac_top
    vacuum_system = vac_system_generator.create_system(vacuum_topology)

    return vacuum_system, vacuum_positions, vacuum_topology, solvated_system, solvated_positions, solvated_topology


smi = 'CCC'
vac_sys, vac_pos, vac_top, solv_sys, solv_pos, solv_top = build_solvated_model(smi)

ress = [res for res in solv_top.residues() if res.name == 'MOL']
assert len(ress) == 1
for atom in ress[0].atoms():
    print(atom.index, atom.name)

os.system('rm -r test')

inter = InterpolAIS(complex_system=solv_sys,
                    complex_topology=solv_top,
                    ligand_system=vac_sys,  # note this must be unique
                    ligand_subset_indices=range(11),
                    temperature=300 * unit.kelvin,
                    pressure=1 * unit.atmosphere,
                    trajectory_directory='test',
                    trajectory_prefix='trial_1',
                    atom_selection_string='all',
                    lambda_sequence=np.zeros(100),
                    ani_model=torchani.models.ANI1ccx(),
                    platform='cpu')

print(solv_pos)
solv_pos = inter.minimize(solv_pos)
print(solv_pos)
inter.anneal(complex_positions=solv_pos,
             n_steps_per_propagation=1,
             gamma=1. / unit.picoseconds,
             dt=1. * unit.femtoseconds,
             metropolize_propagator=False)
