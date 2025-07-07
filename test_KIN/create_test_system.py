import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

os.environ["AMUSE_CHANNELS"] = "local"
from amuse.units import units, constants, nbody_system # type: ignore
from amuse.lab import Particles, Particle # type: ignore

arbeit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(arbeit_path))

def create_test_star(M = 1):
    '''Creates a particle representing a star'''
    star = Particle()
    star.name = 'star'
    star.mass = M | units.Msun

    star.position = (0, 0, 0) | units.AU 
    star.velocity = (0, 0, 0) | units.AU / units.day

    return star

def relative_orbital_velocity(mass, distance):
    return (constants.G * mass / distance).sqrt()

def create_test_planet(sys, M, a, name = 'planet', phase = 0):
    '''creates a test planet with mass M in solar masses, at semi-major axis a in AU 
    from the center of the system.'''
    planet = Particle()
    planet.name = name
    planet.mass = M | units.Msun
    planet.position = (a * np.cos(phase), a * np.sin(phase), 0) | units.AU

    orb_vel = relative_orbital_velocity(sys.total_mass(), planet.position.length())
    # By phase, we mean how far along the orbit the planet is at the start of the simulation.
    planet.velocity = (orb_vel * np.sin(phase), -orb_vel * np.cos(phase), 0 | units.AU / units.day)
    
    return planet

def create_test_system(M_maj = 1e-3, M_min = 1e-5, a_maj = 10, a_min = 1, phaseseed = 0):
    '''Creates a test system with 2 planets: a major and a minor. The central body has a 
    mass of 1 solar mass'''
    phaseseed = int(phaseseed)
    r1 = np.random.default_rng(phaseseed)

    sys = Particles()
    star = create_test_star(M = 1) # change if you like 
    sys.add_particle(star)
    sys.phaseseed = phaseseed # add phaseseed to the system to identify it

    pl_major = create_test_planet(sys, M_maj, a_maj, name = 'major', phase = r1.uniform(0, 2 * np.pi))
    sys.add_particle(pl_major)
    pl_minor = create_test_planet(sys, M_min, a_min, name = 'minor', phase = r1.uniform(0, 2 * np.pi))
    sys.add_particle(pl_minor)

    sys.move_to_center()
    return sys

def test_generation():
    from Trappist.t_plotting import plot_system
    test_sys = create_test_system(M_maj = 1e-3, 
                                  M_min = 1e-5, 
                                  a_maj = 10,
                                  a_min = 1)
    print(test_sys)
    plot_system(test_sys, title='test_system')

def generate_your_system():
    '''Edit this function to generate a system according to your needs. A few examples are provided below.
    This function should return an AMUSE Particles object and the system type (as a string),
    which will be used when saving the results.'''

    ### YOUR GENERATION CODE HERE ###
    sys = Particles()
    sys_type = 'user_type'
    return sys, sys_type

def generate_your_system(): # SIMPLE 3 BODY VERSION
    M_maj = 1e-3
    M_min = 1e-5
    a_maj = 10
    a_min = 20
    phaseseed = 42
    sys_type = '3_bodies'
    return create_test_system(M_maj, M_min, a_maj, a_min, phaseseed), sys_type

# def generate_your_system(): # MANY BODIES VERSION
#     # TODO: where you have lots of bodies bla bla bla. 
#     sys_type = 'many_bodies'
#     return sys, sys_type

# def generate_your_system(): # TRAPPIST VERSION
#     from Trappist.generate_trappist import create_trappist_system
#     phaseseed = 42
#     sys_type = 'Trappist'
#     return create_trappist_system(phaseseed), sys_type

# def generate_your_system(): # S STAR VERSION
#     from S_Stars.generate_s_stars import create_s_star_system
#     filepath = arbeit_path / 'Data/SStars2009ApJ692_1075GTab7.h5'
#     s_star_mass = 20 # dummy mass of all s stars, in solar masses
#     ref_time = 0 | units.yr # reference time for calculation of mean anomaly
#     sys_type = 's_stars'
#     return create_s_star_system(filepath, s_star_mass, ref_time), sys_type

# def generate_your_system(): # SOLAR SYSTEM VERSION
#     from amuse.ext.solarsystem import new_solar_system #type: ignore
#     solarsys = new_solar_system()
#     sys_type = 'solarsystem'
#     return solarsys, sys_type

def plot_evolution_example():
    from Trappist.evolve_trappist import evolve_sys_sakura
    from Trappist.t_plotting import plot_system
    test_sys, sys_type = generate_your_system()
    sys, pos_states, vel_states, total_energy, evolve_time = evolve_sys_sakura(
                        sys = test_sys, 
                        evolve_time = 2400 | units.day,
                        tau_ev = 0.1 | units.day,
                        cache = False
    )
    plot_system(sys, pos_states, vel_states, save = True, filename = f'evolved_{sys_type}.png')

plot_evolution_example()

# generate_your_system()




    
