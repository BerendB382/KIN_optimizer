import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["AMUSE_CHANNELS"] = "local"
from amuse.units import units, constants, nbody_system # type: ignore
from amuse.community.sakura.interface import Sakura # type: ignore

import pickle as pkl
from hashlib import sha256

from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent

def generate_cache_key(particles, evolve_time, tau_ev, mode = 'trappist'):
    '''Generates a unique key for the cache based on the particles, evolve time and evolution time step'''
    key = sha256()
    if mode == 'trappist':
        key.update(pkl.dumps(particles.phaseseed))
        key.update(pkl.dumps(particles.mass))
    if mode == 's_stars':
        key.update(pkl.dumps(particles.s_star_mass))
    key.update(pkl.dumps(evolve_time))
    key.update(pkl.dumps(tau_ev))
    return key.hexdigest()

def get_cache_file(particles, evolve_time, tau_ev, base_cache_dir = root_dir / 'zCache'):
    '''Gets the cache file for the given particles, evolve time and evolution time step'''
    base_cache_dir = Path(base_cache_dir).expanduser()
    base_cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = generate_cache_key(particles, evolve_time, tau_ev)
    cache_file = base_cache_dir / f"{cache_key}.pkl"

    return cache_file

def evolve_sys_sakura(sys, evolve_time, tau_ev, cache = True, print_progress = False):
    '''Evolve a system a certain evolve_time using a time step tau_ev. Uses the Sakura integrator.
    When the simulation is finished, creates a unique cache file identified by
    the system, the evolve time and the timestep size. When the function is called again
    with those same parameters, it loads the result from the associated cache file.
    Use cache = False to turn this caching behavior off and always simulate the system.'''
    # First make sure all units are correct and that the evolve time is a multiple of tau_ev.
    evolve_time = evolve_time.value_in(units.day)
    tau_ev = tau_ev.value_in(units.day)

    if evolve_time % tau_ev != 0:
        evolve_time = np.ceil(evolve_time / tau_ev) * tau_ev
        print(f'Evolve time is not a multiple of tau_ev. Rounding up to {evolve_time} days.')
    
    if cache: # If cache mode is on, try to retrieve the simulation 
        # Check if a cache file exists. If it does, the simulation has already been run for
        # these parameters. A cache file is identified by a system parameter (such as the phaseseed)
        # , the evolve time, and the timestep. 
        cache_file = get_cache_file(sys, evolve_time, tau_ev)

        if cache_file.exists():
            print(f'Loading result from cache: {cache_file}')
            with cache_file.open('rb') as f:
                sys, pos_states, vel_states, total_energy = pkl.load(f)
        else:
            print('Running simulation...')
            # We have to create a converter for the system so 
            # that Sakura can use nbody units internally.
            converter = nbody_system.nbody_to_si(sys.mass.sum(), sys[1].position.length())
            gravity = Sakura(converter)

            # We add the particles to the gravity code and 
            # create a channel to update the particles in the original system.
            gravity.particles.add_particles(sys)
            channel = gravity.particles.new_channel_to(sys)

            pos_states = np.zeros((int(evolve_time / tau_ev) + 1, len(sys), 3))
            vel_states = np.zeros((int(evolve_time / tau_ev) + 1, len(sys), 3))
            total_energy = np.zeros(int(evolve_time / tau_ev) + 1)

            times = np.arange(0, evolve_time + tau_ev, tau_ev)

            for time_idx, time in enumerate(times):
                gravity.evolve_model(time | units.day)
                channel.copy()

                pos_states[time_idx, :] = sys.position.value_in(units.AU)
                vel_states[time_idx, :] = sys.velocity.value_in(units.AU / units.day)
                total_energy[time_idx] = sys.kinetic_energy().value_in(units.J) + sys.potential_energy(G = constants.G).value_in(units.J)
                if print_progress:
                    print(f'Time: {(time | units.day).value_in(units.yr)} years. Energy: {total_energy[time_idx]}')

            gravity.stop()
            
            with cache_file.open('wb') as f:
                pkl.dump([sys, pos_states, vel_states, total_energy], f)
            print(f'Saved result to cache: {cache_file}')

    else: # if caching mode is not on, just run the simulation like normal
        print('Running simulation...')
        # We have to create a converter for the system so 
        # that Sakura can use nbody units internally.
        converter = nbody_system.nbody_to_si(sys.mass.sum(), sys[1].position.length())
        gravity = Sakura(converter)

        # We add the particles to the gravity code and 
        # create a channel to update the particles in the original system.
        gravity.particles.add_particles(sys)
        channel = gravity.particles.new_channel_to(sys)

        pos_states = np.zeros((int(evolve_time / tau_ev) + 1, len(sys), 3))
        vel_states = np.zeros((int(evolve_time / tau_ev) + 1, len(sys), 3))
        total_energy = np.zeros(int(evolve_time / tau_ev) + 1)

        times = np.arange(0, evolve_time + 1*tau_ev, tau_ev)

        for time_idx, time in enumerate(times):
            gravity.evolve_model(time | units.day)
            channel.copy()

            pos_states[time_idx, :] = sys.position.value_in(units.AU)
            vel_states[time_idx, :] = sys.velocity.value_in(units.AU / units.day)
            total_energy[time_idx] = sys.kinetic_energy().value_in(units.J) + sys.potential_energy(G = constants.G).value_in(units.J)
            if print_progress:
                print(f'Time: {(time | units.day).value_in(units.yr)} years. Energy: {total_energy[time_idx]}')

        gravity.stop()
        
    return sys, pos_states, vel_states, total_energy, evolve_time

def test_evolution():
    from generate_trappist import create_trappist_system
    from t_plotting import create_sys_movie, plot_system
    
    test_sys = create_trappist_system()

    evolve_time = 15 | units.day
    tau_ev = 0.15 | units.day

    # plot_system(test_sys, [0], [0])
    evolved_sys, pos_states, vel_states, total_energy, evolve_time = evolve_sys_sakura(sys = test_sys,
                                                                          evolve_time = evolve_time,
                                                                          tau_ev = tau_ev,
                                                                          print_progress = False)
    
    plot_system(evolved_sys, pos_states, vel_states)
    create_sys_movie(evolved_sys, pos_states, vel_states, 'test_movie.mp4', three_d = True)
    

# test_evolution()
    







