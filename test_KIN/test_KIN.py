import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
tf.print(f"Running {__file__} as {__name__}")
print('before imports')

from amuse.units import units # type: ignore

import os
import numpy as np 
from pathlib import Path
import h5py
import sys
import argparse
import json


# This file tests the KIN for a given system. It should create a system according to the create_test_system file,
# and take in the parameters in the job_params file. Maybe have a slurm mode? Or just allow one repetition?
# Also point users to the validation file, and maybe streamline that. 
print('top of the file')
# This file should be called from test_KIN.sh
arbeit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(arbeit_path))
print('before function definitions')

def save_single_test_results(
        path, filename,
        finalmasses, true_masses, mass_error, avg_loss_per_epoch,
        params, mass_array
    ):
    filepath = path / filename
    with h5py.File(filepath, 'a') as f:
        exp_index = len(f.keys())
        exp_group = f.create_group(f'exp_{exp_index}')

        # Store array data
        exp_group.create_dataset('finalmasses', data=finalmasses)
        exp_group.create_dataset('true_masses', data=true_masses)
        exp_group.create_dataset('mass_error', data=mass_error)
        exp_group.create_dataset('avg_loss_per_epoch', data=avg_loss_per_epoch)
        exp_group.create_dataset('mass_per_epoch', data=mass_array)

        # Store params dictionary as attributes or datasets
        param_group = exp_group.create_group('params')

        for key, value in params.items():
            # Handle string-type parameters separately since h5py needs explicit string dtype
            if isinstance(value, str):
                param_group.create_dataset(key, data=value, dtype=h5py.string_dtype(encoding='utf-8'))
            else:
                param_group.create_dataset(key, data=value)
    
    print(f'Saved results in {filepath} under group exp_{exp_index}')

    fig, axs = plt.subplots(2, 1, figsize=(10, 15), layout='compressed', sharex=True)

    axs[0].set_facecolor('xkcd:light grey')
    axs[1].set_facecolor('xkcd:light grey')
    axs[0].plot(avg_loss_per_epoch)
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Loss (log)')
    axs[0].set_title('Average log loss per epoch')
    for i in range(mass_array.shape[1]):
        axs[1].plot(mass_array[:, i], label=f'Body {i+1}')
        axs[1].axhline(true_masses[i], linestyle='--', color='black', alpha=0.5)
    axs[1].set_yscale('log')
    axs[1].set_ylabel('Mass (log(M_sun))')
    axs[1].set_xlabel('Epochs')
    axs[1].set_title('Mass evolution per epoch')
    axs[1].legend()

    fig.savefig(f'{filepath}_{exp_index}.png', dpi = 800)

def test_KIN():
    '''Test the Keplerian Integration Network, with parameters set by the test_params.json file.
    The system to test is generated according to a chosen function in the create_test_system.py file.'''
    from create_test_system import generate_your_system # edit this function in the create_test_system.py file
    from Trappist.evolve_trappist import evolve_sys_sakura
    from Trappist.data_conversion import convert_states_to_celestial_bodies, convert_sys_to_initial_guess_list
    from test_Main_Code import init_optimizer
    import Learning.Training_loops as node
    from Validation.validation_funcs import select_masses, calculate_mass_error

    print('parsing arguments...')
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_file", type=str, required=True, help='Path to the parameter file (JSON)')
    parser.add_argument("--job_id", type=str, required=True, help='Run identifier')
    args = parser.parse_args()
    job_id = args.job_id
    # Load parameters from JSON file
    with open(args.param_file, 'r') as f:
        params = json.load(f)
    
    # create directory to store result
    os.makedirs(arbeit_path / f'test_KIN/test_results', exist_ok=True)
    output_path = arbeit_path / f'test_KIN/test_results'

    # generate a system and evolve it 
    print('Generating and evolving system...')
    sys, sys_type = generate_your_system() # edit this function in the create_test_system file.
    evolved_sys, pos_states, vel_states, total_energy, evolve_time2 = evolve_sys_sakura(
        sys=sys,
        evolve_time=params['evolve_time'] | units.day,
        tau_ev=params['tau'] |units.day,
        print_progress=False,
        cache=False
    )

    num_bodies = len(sys)
    initial_guess = np.ones(num_bodies) * params['init_guess_planet'] # create array of initial guesses 
    initial_guess[0] = params['init_guess_central']

    bodies_and_initial_guesses = convert_sys_to_initial_guess_list(evolved_sys, initial_guess)
    
    # create the CelestialBodies object
    test_bodies = convert_states_to_celestial_bodies(
        pos_states=pos_states,
        vel_states=vel_states,
        num_points_considered_in_cost_function=params['num_points_considered_in_cost_function'],
        evolve_time=evolve_time2 | units.day,
        tau_opt=params['tau'],
        bodies_and_initial_guesses=bodies_and_initial_guesses,
        unknown_dimension=params['unknown_dimension'],
        sort_by_mass=False
    )
    
    # Initialize the optimizer with n + n*3*2 variables, for masses,
    # positions and velocities for each body
    if params['optimizer_type'] == 'BT':
        learning_rate_arr = np.ones(shape=num_bodies+num_bodies*3*2) * params['learning_rate']
    else:
        learning_rate_arr = params['learning_rate']

    optimizer = init_optimizer(
        params['optimizer_type'], num_bodies + num_bodies * 3 * 2, lr=learning_rate_arr
    )
    print('Starting learn masses function...')
    masses, losses = node.learn_masses(
        tau=params['tau'],
        optimizer=optimizer,
        availabe_info_of_bodies=test_bodies,
        epochs=params['epochs'],
        unknown_dimension=params['unknown_dimension'],
        plotGraph=False,
        plot_in_2D=True,
        zoombox='not yet', # can be: trappist, TODO add options
        negative_mass_penalty=1,
        accuracy=params['accuracy'],
        printing=True
    )

    finalmasses, best_idx, avg_loss_per_epoch = select_masses(masses, losses, lowest_loss=False)
    true_masses = sys.mass.value_in(units.Msun) 
    masses = np.array(masses)
    mass_error = calculate_mass_error(finalmasses, sys, relative=True, sumup=False)
    
    print('---------RESULTS---------')
    print('KIN found these masses:\n', finalmasses)
    print('True masses were:\n', true_masses)
    print('Relative errors per body:\n', mass_error)

    save_single_test_results(
        path=output_path, filename=f'{sys_type}_{job_id}.h5',
        finalmasses=finalmasses, true_masses=true_masses,
        mass_error=mass_error, avg_loss_per_epoch=avg_loss_per_epoch,
        params=params, mass_array=masses
    )

if __name__ == "__main__":
    print('starting test_KIN function')
    test_KIN()

    
