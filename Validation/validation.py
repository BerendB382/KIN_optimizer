import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
from pathlib import Path
import os

os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "true"
from amuse.units import units, constants, nbody_system # type: ignore
from amuse.lab import Particles, Particle # type: ignore

arbeit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(arbeit_path))
plot_path = arbeit_path / 'Plots'

def find_masses(
        test_sys, evolve_time, tau_ev, tau_opt, num_points_considered_in_cost_function, 
        unknown_dimension=3, learning_rate=1e-5, init_guess_offset=1e-7, epochs=100, 
        accuracy=1e-8, optimizer_type='ADAM', printing=True
        ):
    '''Find the masses of a test system by evolving it for a certain time with a certain timestep, converting the states
    inbetween to a CelestialBodies object that the learn_masses function can use.'''
    from Trappist.evolve_trappist import evolve_sys_sakura
    from Trappist.data_conversion import convert_states_to_celestial_bodies, convert_sys_to_initial_guess_list
    from test_Main_Code import init_optimizer
    import Learning.Training_loops as node

    original_sys = copy.deepcopy(test_sys)
    if printing:
        print('evolve the system')
    evolved_sys, pos_states, vel_states, total_energy, evolve_time2 = evolve_sys_sakura(
        sys=test_sys,
        evolve_time=evolve_time,
        tau_ev=tau_ev,
        print_progress=False,
        cache=False
        )
    
    # Add some random offset to the initial guesses
    num_bodies = len(test_sys)
    init_guess_variance = np.random.uniform(0, init_guess_offset, num_bodies)
    init_guess_variance[0] = 0
    initial_guess = evolved_sys.mass + (init_guess_variance | units.Msun)

    bodies_and_initial_guesses = convert_sys_to_initial_guess_list(evolved_sys, initial_guess.value_in(units.Msun))

    test_bodies = convert_states_to_celestial_bodies(
        pos_states=pos_states,
        vel_states=vel_states,
        num_points_considered_in_cost_function=num_points_considered_in_cost_function,
        evolve_time=evolve_time2 | units.day,
        tau_opt=tau_opt.value_in(units.day),
        bodies_and_initial_guesses=bodies_and_initial_guesses,
        unknown_dimension=unknown_dimension,
        sort_by_mass=False
        )
    
    # Initialize the optimizer with n + n*3*2 variables, for masses, 
    # velocities and positions for each body
    if optimizer_type == 'BT':
        learning_rate_arr = np.ones(shape=num_bodies+num_bodies*3*2) * learning_rate
    else:
        learning_rate_arr = learning_rate
    # learning_rate_arr[0] = 0

    optimizer = init_optimizer(
        optimizer_type, num_bodies + num_bodies * 3 * 2, lr=learning_rate_arr)
    print('start the learn masses function')
    masses, losses = node.learn_masses(
        tau=tau_opt.value_in(units.day), optimizer=optimizer,
        availabe_info_of_bodies=test_bodies,
        epochs=epochs,
        unknown_dimension=unknown_dimension,
        plotGraph=False,
        plot_in_2D=False,
        zoombox='not yet', # can be 'trappist' 
        negative_mass_penalty=1,
        accuracy=accuracy,
        printing=printing
    )

    return masses, losses

def find_masses_pv_unc(
        test_sys, evolve_time, tau_ev, tau_opt, num_points_considered_in_cost_function, 
        unknown_dimension=3, learning_rate=1e-5, init_guess_offset=1e-7, epochs=100, 
        accuracy=1e-8, optimizer_type='Adam', pv_unc=None, printing=True
        ):
    '''Find the masses of a test system by evolving it for a certain time with a certain timestep, converting the states
    inbetween to a CelestialBodies object that the learn_masses function can use.'''
    from Trappist.evolve_trappist import evolve_sys_sakura
    from Trappist.data_conversion import convert_states_to_celestial_bodies, convert_sys_to_initial_guess_list
    from test_Main_Code import init_optimizer
    import Learning.Training_loops as node

    original_sys = copy.deepcopy(test_sys)
    if printing:
        print('evolve the system')
    evolved_sys, pos_states, vel_states, total_energy, evolve_time2 = evolve_sys_sakura(
        sys=test_sys,
        evolve_time=evolve_time,
        tau_ev=tau_ev,
        print_progress=False,
        cache=False
        )
    
    # Add some random offset to the initial guesses
    num_bodies = len(test_sys)
    init_guess_variance = np.random.uniform(0, init_guess_offset, num_bodies)
    init_guess_variance[0] = 0
    initial_guess = evolved_sys.mass + (init_guess_variance | units.Msun)

    bodies_and_initial_guesses = convert_sys_to_initial_guess_list(evolved_sys, initial_guess.value_in(units.Msun))
    
    pos_variance = np.random.normal(0, pv_unc[0], size=pos_states.shape)
    vel_variance = np.random.normal(0, pv_unc[1], size=vel_states.shape)

    pos_states += pos_variance
    vel_states += vel_variance

    test_bodies = convert_states_to_celestial_bodies(
        pos_states=pos_states,
        vel_states=vel_states,
        num_points_considered_in_cost_function=num_points_considered_in_cost_function,
        evolve_time=evolve_time2 | units.day,
        tau_opt=tau_opt.value_in(units.day),
        bodies_and_initial_guesses=bodies_and_initial_guesses,
        unknown_dimension=unknown_dimension,
        sort_by_mass=False
        )
    
    if optimizer_type == 'BT':
        learning_rate_arr = np.ones(shape=num_bodies+num_bodies*3*2) * learning_rate
    else:
        learning_rate_arr = learning_rate
    # learning_rate_arr[0] = 0

    optimizer = init_optimizer(
        optimizer_type, num_bodies + num_bodies * 3 * 2, lr=learning_rate_arr)
    print('start the learn masses function')
    masses, losses = node.learn_masses(
        tau=tau_opt.value_in(units.day), optimizer=optimizer,
        availabe_info_of_bodies=test_bodies,
        epochs=epochs,
        unknown_dimension=unknown_dimension,
        plotGraph=False,
        plot_in_2D=False,
        zoombox='not yet', # can be 'trappist' 
        negative_mass_penalty=1,
        accuracy=accuracy,
        printing=printing
    )

    return masses, losses

def process_single_system_mp_pv_unc(
        M_min, a_min, evolve_time, tau, num_points_considered_in_cost_function,
        M_maj, a_maj, epochs, accuracy, n_samples, init_guess_offset,
        learning_rate, unknown_dimension, phaseseed, results_path, i,
        optimizer_type, varied_param_names, varied_params, pv_unc
        ):
    from Validation.system_generation import create_test_system
    from Validation.validation_funcs import select_masses, calculate_mass_error, save_results
    # First, generate a system according to the parameters
    test_sys = create_test_system(M_maj=M_maj, M_min=M_min, a_maj=a_maj, a_min=a_min, phaseseed=phaseseed)

    # Find the masses of the system
    masses, losses = find_masses_pv_unc(
        test_sys=test_sys,
        evolve_time=evolve_time | units.day,
        tau_ev=tau | units.day,
        tau_opt=tau | units.day,
        num_points_considered_in_cost_function=num_points_considered_in_cost_function,
        unknown_dimension=unknown_dimension,
        learning_rate=learning_rate,
        init_guess_offset=init_guess_offset,
        epochs=epochs,
        accuracy=accuracy,
        optimizer_type=optimizer_type,
        pv_unc=pv_unc,
        printing=False
        )
    
    masses, best_idx, avg_loss_per_epoch = select_masses(masses, losses, lowest_loss=False)
    true_masses = test_sys.mass.value_in(units.Msun)

    # Calculate the mass error
    mass_error = calculate_mass_error(masses, test_sys, relative=True, sumup=False) # now gives error per planet

    save_results(
        results_path, f'{i}_of_{n_samples:03d}_systems.h5', masses, 
        true_masses, mass_error, avg_loss_per_epoch, 
        varied_param_names, varied_params, pv_unc
        )
    
    print(f'saved results for system {i}')

def process_single_system_mp(
        M_min, a_min, evolve_time, tau, num_points_considered_in_cost_function,
        M_maj, a_maj, epochs, accuracy, n_samples, init_guess_offset,
        learning_rate, unknown_dimension, phaseseed, results_path, i,
        optimizer_type, varied_param_names, varied_params
        ):
    from Validation.system_generation import create_test_system
    from Validation.validation_funcs import select_masses, calculate_mass_error, save_results
    
    # First, generate a system according to the parameters
    test_sys = create_test_system(M_maj=M_maj, M_min=M_min, a_maj=a_maj, a_min=a_min, phaseseed=phaseseed)

    # Find the masses of the system
    masses, losses = find_masses(
        test_sys=test_sys,
        evolve_time=evolve_time | units.day,
        tau_ev=tau | units.day,
        tau_opt=tau | units.day,
        num_points_considered_in_cost_function=num_points_considered_in_cost_function,
        unknown_dimension=unknown_dimension,
        learning_rate=learning_rate,
        init_guess_offset=init_guess_offset,
        epochs=epochs,
        accuracy=accuracy,
        optimizer_type=optimizer_type,
        printing=False
        )
    
    masses, best_idx, avg_loss_per_epoch = select_masses(masses, losses, lowest_loss=False)
    true_masses = test_sys.mass.value_in(units.Msun)

    # Calculate the mass error
    mass_error = calculate_mass_error(masses, test_sys, relative=True, sumup=False) # now gives errors for each body

    save_results(results_path, f'{i}_of_{n_samples:03d}_systems.h5', masses, true_masses, mass_error, avg_loss_per_epoch, 
                 varied_param_names, varied_params)
    print(f'saved results for system {i}')

def test_many_systems(
        M_min, a_min, evolve_time, tau, num_points_considered_in_cost_function,
        M_maj, a_maj, epochs, accuracy, n_samples, init_guess_offset,
        learning_rate, unknown_dimension, phaseseed, optimizer_type, hypercube_state,
        loglog, p_unc = None, v_unc = None, job_id = None,
        ):
    from Validation.validation_funcs import get_latin_sample, merge_h5_files, process_result
    from multiprocessing import Pool
    import h5py

    # Replace by own job id if not using slurm
    if job_id == None:
        job_id = os.environ['SLURM_JOB_ID']

    # set the right paths
    os.makedirs(arbeit_path / f'Validation/val_results/{job_id}', exist_ok=True)
    os.makedirs(arbeit_path / f'Validation/val_results/{job_id}/mp_results', exist_ok=True)
    results_path = arbeit_path / f'Validation/val_results/{job_id}/mp_results' # save the temporary h5 files per system here
    output_path = arbeit_path / f'Validation/val_results/{job_id}' # and combine them here.

    # All this is for automatically detecting what parameters we want to change and logging that
    # in the output file. Looks a bit busy but the end result should be nice!
    param_name_list = [
        'M_min', 'a_min', 'evolve_time', 'tau', 'num_points_considered_in_cost_function',
        'M_maj', 'a_maj', 'epochs', 'accuracy', 'n_samples', 'init_guess_offset',
        'learning_rate', 'unknown_dimension', 'phaseseed', 'hypercube_state',
        'loglog'
        ]
    param_list = [
        M_min, a_min, evolve_time, tau, num_points_considered_in_cost_function,
        M_maj, a_maj, epochs, accuracy, n_samples, init_guess_offset,
        learning_rate, unknown_dimension, phaseseed, hypercube_state,
        loglog
        ]
    
    # we don't allow the user to vary these, as they don't make sense to vary
    unvariable = ['epochs', 'accuracy', 'n_samples', 'unknown_dimension', 'phaseseed', 'hypercube_state', 'loglog']

    # initialize arrays to store which parameters to vary
    param_dict = {}
    p_var_bounds = []
    p_var_names = []

    # store the parameters in a dictionary and select the ones to vary
    for i in range(len(param_list)):
        param_dict[f'{param_name_list[i]}'] = param_list[i]
        if type(param_dict[f'{param_name_list[i]}']) is list:
            if param_name_list[i] in unvariable:
                raise Exception('Not allowed to vary this parameter. Variable parameters are:\n M_min, a_min, evolve_time, tau, num_points_considered_in_cost_function, M_maj, a_maj, init_guess_offset')
            p_var_bounds.append(param_dict[f'{param_name_list[i]}'])
            p_var_names.append(param_name_list[i])
    if len(p_var_bounds) == 0:
        p_v_uncertainty = [p_unc, v_unc]

        output_filename = f'{n_samples}_systems_pv_unc_{job_id}.h5'
        output_file = output_path / output_filename

        # Write the parameters to the output file for future reference
        with h5py.File(output_file, 'w') as f:
            for i, attribute in enumerate(param_list):
                f.attrs[f'{param_name_list[i]}'] = attribute
            f.attrs['varied_param_names'] = ['p_unc', 'v_unc']
            f.attrs['p_v_uncertainty'] = p_v_uncertainty

        # Sample the errors in a latin hypercube sort of way
        p_unc_bounds = [1e-20, p_unc]
        v_unc_bounds = [1e-20, v_unc]
        unc_array = get_latin_sample(n_samples, p_unc_bounds, v_unc_bounds, hypercube_state, loglog)
        
        # Prepare the arguments for the process_single_system_mp function
        args = []
        for i, param_value in enumerate(unc_array):
            # Use the existing param_dict to dynamically set the two parameters being varied
            param_dict_copy = param_dict.copy()
            param_dict_copy['results_path'] = results_path
            param_dict_copy['i'] = i
            param_dict_copy['optimizer_type'] = optimizer_type
            param_dict_copy['varied_param_names'] = []
            param_dict_copy['varied_params'] = []
            param_dict_copy['pv_unc'] = unc_array[i]

            # Append the arguments as a tuple for starmap
            args.append(tuple(param_dict_copy[param] for param in [
                'M_min', 'a_min', 'evolve_time', 'tau', 'num_points_considered_in_cost_function',
                'M_maj', 'a_maj', 'epochs', 'accuracy', 'n_samples', 'init_guess_offset',
                'learning_rate', 'unknown_dimension', 'phaseseed', 'results_path', 'i',
                'optimizer_type', 'varied_param_names', 'varied_params', 'pv_unc'
            ]))
        
        # Now run the tests!
        if job_id.isdigit():
            n_cores = os.environ['SLURM_CPUS_PER_TASK']
            print('n_cores available for slurm is', n_cores)
        else: 
            n_cores = os.cpu_count()
        
        with Pool(processes=int(n_cores)) as pool:
            pool.starmap(process_single_system_mp_pv_unc, args)

    elif len(p_var_bounds) > 2:
        raise Exception(f'Trying to vary {len(p_var_bounds)} parameters. Maximum is 2.')

    elif len(p_var_bounds) == 1:
        # vary 1 parameter.
        p_var_bounds = p_var_bounds[0]
        output_filename = f'{n_samples}_systems_{p_var_names[0]}_{job_id}.h5'
        output_file = output_path / output_filename

        # write the parameters to the output file for future reference
        with h5py.File(output_file, 'w') as f:
            for i, attribute in enumerate(param_list):
                f.attrs[f'{param_name_list[i]}'] = attribute
            f.attrs['varied_param_names'] = p_var_names
        
        # evenly sample the parameter space. 
        if loglog:
            param_sample = 10**(np.linspace(np.log10(p_var_bounds[0]), np.log10(p_var_bounds[1]), n_samples))
        else:
            param_sample = np.linspace(p_var_bounds[0], p_var_bounds[1], n_samples)
        
        # if the parameter to be changed is the number of cost function points, it must be an integer. 
        # fix that here
        if p_var_names[0] == 'num_points_considered_in_cost_function':
            param_sample = np.ceil(param_sample).astype(int)
        
        # prepare the arguments for the process_single_system_mp function
        args = []
        for i, param_value in enumerate(param_sample):
            # use the existing param_dict to dynamically set the two parameters being varied
            param_dict_copy = param_dict.copy()
            param_dict_copy[p_var_names[0]] = param_value
            param_dict_copy['results_path'] = results_path
            param_dict_copy['i'] = i
            param_dict_copy['optimizer_type'] = optimizer_type
            param_dict_copy['varied_param_names'] = p_var_names
            param_dict_copy['varied_params'] = param_value
            
            # append the arguments as a tuple for starmap
            args.append(tuple(param_dict_copy[param] for param in [
                'M_min', 'a_min', 'evolve_time', 'tau', 'num_points_considered_in_cost_function',
                'M_maj', 'a_maj', 'epochs', 'accuracy', 'n_samples', 'init_guess_offset',
                'learning_rate', 'unknown_dimension', 'phaseseed', 'results_path', 'i',
                'optimizer_type', 'varied_param_names', 'varied_params'
            ]))
        
        # now run the tests!
        if job_id.isdigit():
            n_cores = os.environ['SLURM_CPUS_PER_TASK']
            print('n_cores available for slurm is', n_cores)
        else: 
            n_cores = os.cpu_count()
        print('starting pools')
        with Pool(processes=int(n_cores)) as pool:
            pool.starmap(process_single_system_mp, args)
        
    elif len(p_var_bounds) == 2:
        # vary 2 parameters.
        output_filename = f'{n_samples}_systems_{p_var_names[0]}_{p_var_names[1]}_{job_id}.h5'
        output_file = output_path / output_filename

        # write the parameters to the output file for future reference
        with h5py.File(output_file, 'w') as f:
            for i, attribute in enumerate(param_list):
                f.attrs[f'{param_name_list[i]}'] = attribute
            f.attrs['varied_param_names'] = p_var_names
        
        # sample the two parameters efficiently using latin hypercube sampling
        param_sample = get_latin_sample(n_samples, p_var_bounds[0], p_var_bounds[1], hypercube_state, loglog)

        # if the parameter to be changed is the number of cost function points, it must be an integer. 
        # fix that here
        
        if p_var_names[0] == 'num_points_considered_in_cost_function':
            param_sample[:, 0] = np.ceil(param_sample[:, 0]).astype(int)
        if p_var_names[1] == 'num_points_considered_in_cost_function':
            param_sample[:, 1] = np.ceil(param_sample[:, 1]).astype(int)

        # prepare the arguments for the process_single_system_mp function
        args = []
        for i, param_values in enumerate(param_sample):
            # use the existing param_dict to dynamically set the two parameters being varied
            param_dict_copy = param_dict.copy()
            param_dict_copy[p_var_names[0]] = param_values[0]
            param_dict_copy[p_var_names[1]] = param_values[1]
            param_dict_copy['results_path'] = results_path
            param_dict_copy['i'] = i
            param_dict_copy['optimizer_type'] = optimizer_type
            param_dict_copy['varied_param_names'] = p_var_names
            param_dict_copy['varied_params'] = param_values

            # append the arguments as a tuple for starmap
            args.append(tuple(param_dict_copy[param] for param in [
                'M_min', 'a_min', 'evolve_time', 'tau', 'num_points_considered_in_cost_function',
                'M_maj', 'a_maj', 'epochs', 'accuracy', 'n_samples', 'init_guess_offset',
                'learning_rate', 'unknown_dimension', 'phaseseed', 'results_path', 'i',
                'optimizer_type', 'varied_param_names', 'varied_params'
            ]))

        # now run the tests!
        n_cores = os.environ['SLURM_CPUS_PER_TASK']
        print('n_cores available for slurm is', n_cores)
        with Pool(processes=int(n_cores)) as pool:
            pool.starmap(process_single_system_mp, args)
    
import argparse
import json

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--param_file", type=str, required=True, help='Path to the parameter file (JSON)')
args = parser.parse_args()

# Load parameters from JSON file
with open(args.param_file, 'r') as f:
    params = json.load(f)

test_many_systems(**params)

# if __name__ == '__main__':
#     test_many_systems(
#         M_min=1e-5, # in solar masses
#         a_min=20, # in AU
#         evolve_time=1200, # in days
#         tau=2, # in days
#         num_points_considered_in_cost_function=4, 
#         M_maj=1e-3, # in solar masses
#         a_maj=10, # in AU
#         epochs=1,
#         accuracy=1e-10,
#         n_samples=8,
#         init_guess_offset=[1e-10, 1e-2], # in solar masses
#         learning_rate=1e-2,
#         unknown_dimension=3,
#         phaseseed=0,
#         optimizer_type='ADAM',
#         hypercube_state=42,
#         loglog=True,
#         p_unc=0.001, # in AU
#         v_unc=0.00001, # in AU/day
#         job_id='testbert'
#     )


    


