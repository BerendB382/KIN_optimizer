import numpy as np
import matplotlib.pyplot as plt

from amuse.units import units, constants, nbody_system # type: ignore
from amuse.lab import Particles, Particle # type: ignore

def load_trappist_params():
    '''Finds the parameters of the TRAPPIST-1 system from the NASA Exoplanet Archive'''
    # This function gives an EOF error the first time you run it.
    # When the data is downloaded, the error disappears.
    from PyAstronomy import pyasl
    nexa = pyasl.ExoplanetsOrg()

    trappist_names = np.array(['TRAPPIST-1 b', 'TRAPPIST-1 c', 'TRAPPIST-1 d', 'TRAPPIST-1 e', 'TRAPPIST-1 f', 'TRAPPIST-1 g', 'TRAPPIST-1 h'])
    trappist_params = []

    for planet in trappist_names:
        planet_params = nexa.selectByPlanetName(planet)
        trappist_params.append(planet_params)

    return np.array(trappist_params)

def create_trappist_star(pl_dat):
    '''Creates a particle representing the TRAPPIST-1 star'''
    trap_star = Particle()
    trap_star_name = pl_dat['pl_hostname']
    trap_star.mass = pl_dat['st_mass'] | units.MSun

    trap_star.position = (0, 0, 0) | units.AU 
    trap_star.velocity = (0, 0, 0) | units.AU / units.day

    return trap_star

def incline_orbit(pos, vel, incl):
    '''Applies an inclination about the y-axis (left-right when looking straight onto the system).
    Rotates both the position vector and the velocity vector {param}:incl degrees. '''
    incl_rad = incl * np.pi / 180
    rot_matrix = np.array([[np.cos(incl_rad), 0, np.sin(incl_rad)],
                           [0, 1, 0],
                           [-np.sin(incl_rad), 0, np.cos(incl_rad)]])

    pos = np.dot(rot_matrix, pos)
    vel = np.dot(rot_matrix, vel)

    return pos, vel

def create_trappist_planet(dat, phase = 0):
    '''Creates a particle representing a TRAPPIST-1 planet'''
    trap_star = create_trappist_star(dat)
    trap_planet = Particle()

    trap_planet_name = dat['pl_name']
    trap_planet.mass = (dat['pl_massj'] | units.MJupiter)
    orb_vel = (2* np.pi * dat['pl_orbsmax']) / (dat['pl_orbper']) | units.AU / units.day

    # Take into account the orbital inclination and the phase the planet starts at.
    # By phase, we mean how far along the orbit the planet is at the start of the simulation.
    trap_planet.position = (dat['pl_orbsmax'] * np.cos(phase), \
                            dat['pl_orbsmax'] * np.sin(phase), \
                            np.random.uniform(-0.001, 0.001)) | units.AU
    trap_planet.velocity = (orb_vel * np.sin(phase), \
                            -orb_vel * np.cos(phase), \
                            np.random.uniform(-0.000001, 0.000001) | units.AU / units.day)
    
    trap_planet.position, trap_planet.velocity = incline_orbit(trap_planet.position, trap_planet.velocity, dat['pl_orbincl'])
    return trap_planet

def create_trappist_system(phaseseed = 0):
    '''Creates a system of particles representing the TRAPPIST-1 system'''
    np.random.seed(phaseseed)

    dat = load_trappist_params()
    sys = Particles()
    trap_star = create_trappist_star(dat[0])
    sys.add_particle(trap_star)
    sys.phaseseed = phaseseed

    for i in range(len(dat)):
        sys.add_particle(create_trappist_planet(dat[i], phase = np.random.uniform(0, 2 * np.pi)))
    sys.move_to_center()
    return sys

def test_generation():
    from t_plotting import plot_system
    test_sys = create_trappist_system()
    plot_system(test_sys, [0], [0], dimension = 3)
    print(test_sys)

# test_generation()






