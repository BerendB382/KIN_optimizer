import numpy as np
import matplotlib.pyplot as plt

from amuse.units import units, constants, nbody_system # type: ignore

import matplotlib.animation as animation
from functools import partial

from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
plots_path = root_dir / 'Plots'
# this file should have functions that:
# - generate a movie based on all the states
# - plot the energy difference throughout time

# set matplotlib parameters
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

font = {'family':'serif',
        'weight':'normal',
        'size'  : 22}

plt.rc('font', **font)

def plot_system(sys, pos_states = [0], vel_states = [0], title = 'TRAPPIST-1 System', dimension = 2, save = False, filename = 'system.pdf'):
    '''Plots a system of particles. You can enter the position states
    of the system to plot the past positions of the planets.'''
    fig = plt.figure(figsize=(8, 8), constrained_layout=True)

    if dimension == 2:
        ax = fig.add_subplot(111)
        for i in range(len(sys)):
            plt.scatter(sys[i].x.value_in(units.AU), sys[i].y.value_in(units.AU), s = 100)
            if len(pos_states) > 1:
                    plt.plot(pos_states[:,i,0],
                            pos_states[:,i,1],
                            alpha = 0.3, color = 'black')
            if len(vel_states) > 1:
                plt.quiver(pos_states[-1][i][0], pos_states[-1][i][1],
                           vel_states[-1][i][0], vel_states[-1][i][1])
            ax.set_xlabel('X (AU)')
            ax.set_ylabel('Y (AU)')
    if dimension == 3:
        ax = fig.add_subplot(111, projection = '3d')
        for i in range(len(sys)):
            plt.scatter(sys[i].x.value_in(units.AU), sys[i].y.value_in(units.AU), s = 100)
            if len(pos_states) > 1:
                    plt.plot(pos_states[:, i, 0],
                         pos_states[:, i, 1],
                         pos_states[:, i, 2],
                         alpha = 0.3, color = 'black')
            ax.set_xlabel('AU')
            ax.set_ylabel('AU')
            ax.set_zlabel('AU')

    plt.grid()
    if save:
        plt.savefig(filename, dpi=800)
    else:
        plt.show()

def update_frame(frame, sys, pos_states, vel_states, ax, three_d=False):
    '''Updates the frame in a movie.'''
    ax.clear()

    # Base the frame of the movie on the maximum distance
    # of a planet from the star
    max_dist = np.max(np.linalg.norm(sys.position.value_in(units.AU))) * 0.2
    if three_d:
        ax.set_xlabel('x [AU]')
        ax.set_ylabel('y [AU]')
        ax.set_zlabel('z [AU]')
           
        ax.set_xlim(-max_dist, max_dist)
        ax.set_ylim(-max_dist, max_dist)
        ax.set_zlim(-max_dist, max_dist)

        ax.set_title(f'System, evolved')
        
        for body_idx in range(len(sys)):
            ax.scatter(pos_states[frame, body_idx, 0],
                       pos_states[frame, body_idx, 1],
                       pos_states[frame, body_idx, 2])
            ax.quiver(pos_states[frame, body_idx, 0],
                       pos_states[frame, body_idx, 1],
                       pos_states[frame, body_idx, 2],
                       vel_states[frame, body_idx, 0],
                       vel_states[frame, body_idx, 1],
                       vel_states[frame, body_idx, 2], 
                       length = 0.5,
                       arrow_length_ratio = 0.2)
    else:
        ax.set_xlabel('x [AU]')
        ax.set_ylabel('y [AU]')

        ax.set_xlim(-max_dist, max_dist)
        ax.set_ylim(-max_dist, max_dist)

        ax.set_title(f'System, evolved ')

        for body_idx in range(len(sys)):
            ax.scatter(pos_states[frame, body_idx, 0],
                       pos_states[frame, body_idx, 1])

def create_sys_movie(sys, pos_states, vel_states, filename, three_d = False, movie_path = root_dir / 'zMovies'):
    '''Creates a movie from the positions saved during evolution of a system.
    Input the evolved system, the position states and a file name ending in .mp4.
    You can also set the movie path and if you want it to be 3d or not.'''

    print('Generating movie...')
    
    fig = plt.figure()

    if three_d:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x [AU]')
        ax.set_ylabel('y [AU]')
        ax.set_zlabel('z [AU]')
    else:
        ax = fig.add_subplot(111)
        ax.set_xlabel('x [AU]')
        ax.set_ylabel('y [AU]')
    
    ani = animation.FuncAnimation(fig, partial(update_frame, 
                                               sys = sys,
                                               pos_states = pos_states,
                                               vel_states = vel_states,
                                               ax = ax,
                                               three_d = three_d),
                                               frames = len(pos_states),
                                               interval = 250)
    
    ani.save(movie_path / filename, writer='ffmpeg', fps = 24)

def plot_loss_func(loss_per_epoch, path = plots_path, name = 'loss_per_epoch.pdf', title = 'Loss per epoch'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average loss')
    ax.set_title(title)
    ax.plot(loss_per_epoch)
    ax.set_yscale('log')
    ax.grid()
    file_path = path / name 
    plt.savefig(file_path, dpi = 600)

def test_plotting():
    from generate_trappist import create_trappist_system

    test_sys = create_trappist_system(phaseseed=42)
    plot_system(test_sys, save=True, filename='Trappist_sys.png')

# test_plotting()
