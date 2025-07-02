import numpy as np
import os

os.environ["AMUSE_CHANNELS"] = "local"
from amuse.units import units, constants, nbody_system # type: ignore
from amuse.lab import Particles, Particle # type: ignore

def convert_sys_to_initial_guess_list(sys, initial_guesses):
    true_masses = sys.mass.value_in(units.Msun)
    initial_guesses = initial_guesses
    bodies_and_initial_guesses = [true_masses, initial_guesses]
    return np.array(bodies_and_initial_guesses)

def select_points(pos, vel, num_points_requested):
    '''
    Selects num_points equally spaced points from pos and vel.
    If the requested number does not fit the number of timesteps, it is adjusted.
    '''
    N = pos.shape[0]  # total number of timesteps
    if num_points_requested < 2:
        raise ValueError("num_points_requested must be at least 2 (start and end)")
    max_intervals = N - 1  # number of intervals between points

    # how many steps in between points?
    stride = max_intervals // (num_points_requested - 1)

    if stride == 0:
        stride = 1
        adjusted_num_points = max_intervals + 1
        print(f"Requested {num_points_requested} points is too high for {N} timesteps. "
              f"Using maximum possible number of points: {adjusted_num_points}")
    else:
        adjusted_num_points = (max_intervals // stride) + 1

        if adjusted_num_points != num_points_requested:
            print(f"⚠️ Adjusted number of points from {num_points_requested} to {adjusted_num_points} "
                  f"to fit evenly into {N} timesteps (stride={stride})")

    selected_indices = np.arange(0, stride * (adjusted_num_points - 1) + 1, stride).astype(int)
    print(selected_indices)
    new_pos = pos[selected_indices]
    new_vel = vel[selected_indices]

    return new_pos, new_vel, selected_indices

def convert_states_to_celestial_bodies(pos_states, vel_states, num_points_considered_in_cost_function, evolve_time, tau_opt, bodies_and_initial_guesses, unknown_dimension, sort_by_mass = False):
    '''Converts arrays of position and velocity states, as well as other things,
    into a CelestialBody objects.'''
    import sys
    import os

    if sort_by_mass:
        sorting_idx = np.argsort(bodies_and_initial_guesses, axis = 1, kind='mergesort')[0, ::-1]
        bodies_and_initial_guesses = np.sort(bodies_and_initial_guesses, axis = 1, kind='mergesort')[:, ::-1]
        pos_states = pos_states[:, sorting_idx, :]
        vel_states = vel_states[:, sorting_idx, :]

    # Add the parent directory to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    import Learning.Body_info_class as clas
    import math

    # Set the amount of points to use and make sure the optimizer
    # knows how many steps of size tau_opt to simulate to 
    # new_step_size = evolve_time / num_points_considered_in_cost_function
    # num_points = num_points_considered_in_cost_function + 1
    # num_of_steps_in_do_step = np.ceil(new_step_size.value_in(units.day) / tau_opt)

    pos_states, vel_states, selected_indices = select_points(pos_states, vel_states, num_points_considered_in_cost_function+1)

    timestep_days = tau_opt
    times_in_days = selected_indices * timestep_days
    num_points = len(selected_indices)

    # Adjust pos and vel based on the unknown dimension.
    # The unknown value is ignored later (in the calculate_loss_derivatives function),
    # but for now we add a small random perturbation to it
    if unknown_dimension in [0, 1, 2]:
        pos_states[:, :, unknown_dimension] += np.random.uniform(0.00001, -0.00001)
        vel_states[:, :, unknown_dimension] += np.random.uniform(0.0000001, -0.0000001)

    # Build CelestialBody objects with states over time
    celestial_bodies = []
    for body_index in range(len(bodies_and_initial_guesses[0])):
        body_mass = bodies_and_initial_guesses[1][body_index]

        states = []
        for i in range(num_points):
            # Compute time in units of tau:
            # i-th data point corresponds to i * time_step_in_days from start_date.
            time_in_tau = times_in_days[i] / timestep_days

            if not math.isclose(time_in_tau, round(time_in_tau), rel_tol=1e-10, abs_tol=1e-10):
                print(
                    f"Warning: time_in_tau for step {i} is not an integer. Consider adjusting parameters.")

            # extract position and velocity
            pos = pos_states[i][body_index]
            vel = vel_states[i][body_index]

            states.append(clas.TimeState(
                time = int(round(time_in_tau)),
                position = pos,
                velocity = vel
            ))
        # take the first body's mass to be known
        if body_index == 0:
            celestial_bodies.append(clas.CelestialBody(
            name=f"body_{body_index}",
            mass = bodies_and_initial_guesses[0][body_index],
            states = states
            ))
        else:
            celestial_bodies.append(clas.CelestialBody(
            name=f"body_{body_index}",
            mass = body_mass,
            states = states
            ))

    return celestial_bodies

def test_conversion():
    from generate_trappist import create_trappist_system
    from evolve_trappist import evolve_sys_sakura
    import math
    import matplotlib.pyplot as plt
    from t_plotting import plot_system
    import sys
    from pathlib import Path

    # Set the working directory to 'Arbeit' dynamically
    arbeit_path = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(arbeit_path))
    from Learning.plotting import draw_reference_positions

    test_sys = create_trappist_system()

    evolve_time = 0.5 | units.yr
    tau_ev = 0.1 | units.day

    # plot_system(test_sys, [0], [0])
    evolved_sys, pos_states, vel_states, total_energy = evolve_sys_sakura(sys = test_sys,
                                                                          evolve_time = evolve_time,
                                                                          tau_ev = tau_ev,
                                                                          print_progress = False)
    
    initial_guesses = evolved_sys.mass + (np.random.uniform(-0.0000001, 0.0000001) | units.Msun)

    bodies_and_initial_guesses = convert_sys_to_initial_guess_list(evolved_sys, initial_guesses)

    celestial_bodies = convert_states_to_celestial_bodies(pos_states = pos_states,
                                                          vel_states = vel_states,
                                                          num_points_considered_in_cost_function = 1,
                                                          evolve_time = evolve_time,
                                                          tau_opt = 0.1,
                                                          bodies_and_initial_guesses = bodies_and_initial_guesses,
                                                          unknown_dimension = 3,
                                                          sort_by_mass = True)
    
    plt.semilogy(total_energy/total_energy[0])
    plt.show()
    print(celestial_bodies)
    if not math.isclose(celestial_bodies[3].states[0].position[0], pos_states[0][3][0], rel_tol = 1e-15, abs_tol=1e-15):
        print('The first state in celestial_bodies does not match the first state in pos_states.')

    if not math.isclose(celestial_bodies[3].states[-1].position[0], pos_states[-1][3][0], rel_tol = 1e-15, abs_tol=1e-15):
        print('The last state in celestial_bodies does not match the first state in pos_states.')

    # plot_system(evolved_sys, pos_states, vel_states)

    # TODO: overplot the states in celestial_bodies to check that they match.


# test_conversion()
    