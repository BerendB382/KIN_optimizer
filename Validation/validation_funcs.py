import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import os
from pathlib import Path

from amuse.units import units, constants, nbody_system # type: ignore
from amuse.lab import Particles, Particle # type: ignore

os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "true"

arbeit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(arbeit_path))
plot_path = arbeit_path / 'Plots'

def select_masses(masses, losses, lowest_loss = True):
    # select the masses from the epoch with the lowest loss value
    losses = np.array(losses)
    average_losses = np.sum(losses, axis = 2)
    avg_loss_per_epoch = average_losses[:, -1]

    good_mass_indices = np.array([np.all(np.array(mass_list) > 0) for mass_list in masses])
    valid_indices = np.where(good_mass_indices)[0]

    best_idx = valid_indices[np.argmin(avg_loss_per_epoch[valid_indices])]

    if lowest_loss:
        masses = masses[best_idx]  
        return masses, best_idx, avg_loss_per_epoch
    else:
        masses = masses[-1]
        return masses, best_idx, avg_loss_per_epoch
    
def calculate_mass_error(new_masses, sys):
    return np.sum(abs(new_masses - sys.mass.value_in(units.Msun))/sys.mass.value_in(units.Msun))/len(sys)

def save_results_old(path, filename, M_min, a_min, masses, mass_error, avg_loss_per_epoch):
    import h5py
    metadata = { 
        'experiment_name': f'Run with M_min={M_min}, a_min={a_min}',
        'M_min': M_min,
        'a_min': a_min, 
    }
    filepath = path / filename
    with h5py.File(filepath, 'a') as f:
        exp_index = len(f.keys())
        exp_group = f.create_group(f'exp_{exp_index}')
    
        # store data in the new group
        exp_group.create_dataset('masses', data = masses)
        exp_group.create_dataset('mass_error', data = mass_error)
        exp_group.create_dataset('avg_loss_per_epoch', data = avg_loss_per_epoch)
        parameters = np.array([M_min, a_min])
        exp_group.create_dataset('parameters (M_min, a_min)', data=parameters)

        # store metadata
        for key, value in metadata.items():
            exp_group.attrs[key] = value
        f.close()

def load_result_old(path, filename, filter_outliers=False):
    '''Loads the results of a particular file. Puts it into a dictionary.'''
    import h5py
    filepath = path / filename

    masses_list = []
    mass_error_list = []
    avg_loss_per_epoch_list = []
    parameters_list = []
    with h5py.File(filepath, 'r') as f:
        for exp_name in f.keys():
            exp_group = f[exp_name]
            mass_data = np.array(exp_group['masses'])
            
            # filter out invalid mass data 
            if mass_data.ndim < 1:
                print(f'skipped run {exp_name} due to invalid mass data')
                continue
            
            masses_list.append(mass_data)
            mass_error_data = np.array(exp_group['mass_error'])

            if filter_outliers == True and mass_error_data > 1:
                print(f'skipped system {exp_name} due to high mass error')
                continue

            mass_error_list.append(mass_error_data)
            avg_loss_per_epoch_list.append(np.array(exp_group['avg_loss_per_epoch']))
            parameters_list.append(np.array(exp_group['parameters (M_min, a_min)']))

        # also extract the attributes.
        print('\nRun parameters:')
        for key, value in f.attrs.items():
            print('   {}: {}'.format(key, value))
        print('\n')
        run_params = f.attrs 

    # make sure all loss arrays are of equal length, by repeating the last loss value for the epochs 
    # where the accuracy limit was already reached. 
    maxlength = np.max(np.array([len(i) for i in avg_loss_per_epoch_list]))
    for i, alpe in enumerate(avg_loss_per_epoch_list):
        if len(alpe) < maxlength:
            avg_loss_per_epoch_list[i] = np.pad(
                alpe, 
                (0, maxlength - len(alpe)), 
                mode='edge'
            )

    masses = np.array(masses_list)
    mass_errors = np.array(mass_error_list)
    avg_loss_per_epoch = np.array(avg_loss_per_epoch_list)
    parameters = np.array(parameters_list)

    return {
        'masses': masses,
        'mass_errors': mass_errors,
        'avg_loss_per_epoch': avg_loss_per_epoch,
        'parameters': parameters
    }, run_params

def sensitivity_plot_old(results, filename, maj_param, log_error=False, plot_path=plot_path, loglog=False):
    '''Creates a sensitivity plot for a given set of results.'''
    from scipy.interpolate import LinearNDInterpolator

    mass_errors = results['mass_errors']
    parameters = results['parameters']

    M_min = parameters[:, 0]
    a_min = parameters[:, 1]

    # interpolate between the parameters
    if loglog:
        plt.loglog()
        M_min = np.log10(M_min)
        a_min = np.log10(a_min)
        M_min_space = np.linspace(np.min(M_min), np.max(M_min), 400)
        a_min_space = np.linspace(np.min(a_min), np.max(a_min), 400)
        maj_param = np.log10(maj_param)
    else: 
        M_min_space = np.linspace(np.min(M_min), np.max(M_min), 400)
        a_min_space = np.linspace(np.min(a_min), np.max(a_min), 400)
        

    M_min_grid, a_min_grid = np.meshgrid(M_min_space, a_min_space)

    interp = LinearNDInterpolator((M_min, a_min), mass_errors)
    Mass_errors_i = interp(M_min_grid, a_min_grid)

    cbarlabel = 'Fractional mass error'

    if log_error: # if we want this, change the error to the log of the error.
        Mass_errors_i = np.log10(Mass_errors_i)
        mass_errors = np.log10(mass_errors)
        cbarlabel = 'Log (10) fractional mass error'
        filename = f'log_{filename}'

    plt.figure(figsize=[15, 9])
    plt.set_cmap('viridis')
    plt.pcolormesh(M_min_grid, a_min_grid, Mass_errors_i, shading='auto')
    plt.scatter(M_min, a_min, s=150, color='white')
    plt.scatter(M_min, a_min, s=60, c=mass_errors, label='True simulations')  # Plot input points
    # plot the position of the major planet
    plt.axhline(maj_param[1], linestyle='--', color='white')
    plt.axvline(maj_param[0], linestyle='--', color='white', label = 'Parameters of major planet')
    
    ax = plt.gca()
    ax.set_facecolor('xkcd:light grey')

    plt.xlabel('Minor planet mass (M_sun)')
    plt.ylabel('Minor planet semimajor axis (AU)')
    plt.title('Sensitivity plot for a three-body system with a major planet and a minor planet.')
    plt.legend(loc='lower right')
    plt.colorbar(label=cbarlabel)

    saved_file = plot_path / filename

    plt.savefig(f'{saved_file}.pdf', dpi=800)

def process_result_old(path, filename, maj_param, log_error = False, filter_outliers=False, loglog=False):
    '''Loads results from an h5 file, and then creates an image.'''
    import h5py
    results, run_params = load_result_old(path, filename, filter_outliers=filter_outliers)
    sensitivity_plot_old(results, f'{filename}', maj_param, log_error, plot_path = plot_path, loglog=loglog)
    print(f'file {filename} processed')

def save_results(
            path, filename, masses, true_masses, mass_error, avg_loss_per_epoch, 
            varied_param_names, varied_params
            ):
    import h5py
    filepath = path / filename
    with h5py.File(filepath, 'a') as f:
        exp_index = len(f.keys())
        exp_group = f.create_group(f'exp_{exp_index}')

        # store data in the new group
        exp_group.create_dataset('masses', data = masses)
        exp_group.create_dataset('true_masses', data = true_masses)
        exp_group.create_dataset('mass_error', data = mass_error)
        exp_group.create_dataset('avg_loss_per_epoch', data = avg_loss_per_epoch)
        parameters = np.array(varied_params)
        if len(varied_param_names) == 2:
            exp_group.create_dataset(f'{varied_param_names[0]}, {varied_param_names[1]}', 
                                 data=parameters)
        else: 
            exp_group.create_dataset(f'{varied_param_names},',
                                 data=parameters)

        f.close()

    
def get_latin_sample(n_samples, bounds1, bounds2, hypercube_state, log_space=True):
    from scipy.stats import qmc   
    sampler = qmc.LatinHypercube(d=2, strength=1, rng=hypercube_state)
    sample_unscaled = sampler.random(n=n_samples)
    if log_space:
        log_bounds1 = np.log10(np.array(bounds1))
        log_bounds2 = np.log10(np.array(bounds2))
        sample = qmc.scale(sample_unscaled, [log_bounds1[0], log_bounds2[0]], [log_bounds1[1], log_bounds2[1]])
        sample = 10**(sample)
    else:
        sample = qmc.scale(sample_unscaled, [bounds1[0], bounds2[0]], [bounds2[1], bounds2[1]])
    return sample

def merge_h5_files(input_folder, output_file, delete=False):
    from pathlib import Path
    import h5py
    input_folder = Path(input_folder)
    output_file = Path(output_file)

    h5_files = sorted(input_folder.glob('*.h5'))

    with h5py.File(output_file, 'a') as output_h5:
        for file_index, h5_file in enumerate(h5_files):
            with h5py.File(h5_file, 'r') as input_h5:
                # copy each group from the input file to the output
                for group_name in input_h5.keys():
                    group_path = f'exp_{file_index}'
                    input_h5.copy(group_name, output_h5, name = group_path)
    
    print(f'All files merged into {output_file}')

    if delete:
        for h5_file in h5_files:
            try:
                os.remove(h5_file)
                print(f"Deleted file: {h5_file}")
            except Exception as e:
                print(f"Error deleting file {h5_file}: {e}")
        os.rmdir(input_folder)

def load_result(path, filename, filter_outliers=False):
    '''Loads the results of a particular file. Puts it into a dictionary.'''
    import h5py
    filepath = path / filename

    masses_list = []
    true_masses_list = []
    mass_error_list = []
    avg_loss_per_epoch_list = []
    parameters_list = []

    with h5py.File(filepath, 'r') as f:
        for exp_name in f.keys():
            exp_group = f[exp_name]
            mass_data = np.array(exp_group['masses'])
            
            # filter out invalid mass data 
            if mass_data.ndim < 1:
                print(f'skipped run {exp_name} due to invalid mass data')
                continue
            
            masses_list.append(mass_data)
            mass_error_data = np.array(exp_group['mass_error'])

            if filter_outliers == True and mass_error_data > 1:
                print(f'skipped system {exp_name} due to high mass error')
                continue

            mass_error_list.append(mass_error_data)
            true_masses_list.append(np.array(exp_group['true_masses']))
            avg_loss_per_epoch_list.append(np.array(exp_group['avg_loss_per_epoch']))

            # very stupid, but works. don't add commas to other dataset names!
            for dataset_name in exp_group.keys():
                if ',' in dataset_name:
                    param_names = dataset_name
                    parameters_list.append(np.array(exp_group[dataset_name]))

        # also extract the attributes.
        print('\nRun parameters:')
        for key, value in f.attrs.items():
            print('   {}: {}'.format(key, value))
        print('\n')
        run_params = {key: value for key, value in f.attrs.items()}
        
    # make sure all loss arrays are of equal length, by repeating the last loss value for the epochs 
    # where the accuracy limit was already reached. 
    maxlength = np.max(np.array([len(i) for i in avg_loss_per_epoch_list]))
    for i, alpe in enumerate(avg_loss_per_epoch_list):
        if len(alpe) < maxlength:
            avg_loss_per_epoch_list[i] = np.pad(
                alpe, 
                (0, maxlength - len(alpe)), 
                mode='edge'
            )

    masses = np.array(masses_list)
    true_masses = np.array(true_masses_list)
    mass_errors = np.array(mass_error_list)
    avg_loss_per_epoch = np.array(avg_loss_per_epoch_list)
    parameters = np.array(parameters_list)

    return {
        'masses': masses,
        'true_masses': true_masses,
        'mass_errors': mass_errors,
        'avg_loss_per_epoch': avg_loss_per_epoch,
        f'{param_names}': parameters
    }, run_params

# list of axis labels to use in the plot
param_labels = [
    'Minor planet Mass (Msun)', 'Minor planet orbital period (days)', 
    'Evolve time (days)', 'Tau (days)', 'Cost function points',
    'Major planet mass (Msun)', 'Major planet orbital period (days)', 
    'Initial guess offset (Msun)'
    ]
# list of log axis labels
log_param_labels = [
    'Log minor planet Mass (log10(Msun))', 'Log minor planet orbital period (log(days))', 
    'Log evolve time (log10(days))', 'Log tau (log10(days))', 'log cost function points',
    'Log major planet mass (log10(Msun))', 'Major planet orbital period (log(days))', 
    'Initial guess offset (log10(Msun))'
    ]
# list of all the options varied_param_names can be.
# A bit cumbersome, but a way to match the names we get out of the file
# to a parameter index.
nameslist = [
    'M_min', 'a_min', 'evolve_time', 'tau', 
    'num_points_considered_in_cost_function',
    'M_maj', 'a_maj', 'init_guess_offset'
    ]

def get_orbital_period(M, a):
    '''Takes in Mass in solar masses, and semimajor axis in AU'''
    p = 2*np.pi*np.sqrt((a|units.AU)**3 / (constants.G*(M|units.Msun)))
    p_in_days = np.array([item.value_in(units.day) for item in p])
    return p_in_days

def sensitivity_plot_1param(results, filename, run_params, log_error=True, plot_path=plot_path, loglog=False):
    '''Creates a 1D sensitivity plot for a given set of results.'''

    varied_param_name = run_params['varied_param_names'][0]
    p_index = np.where(np.array(nameslist) == varied_param_name)[0].item()

    mass_errors = results['mass_errors']
    p = results[f'{varied_param_name},']

    if p_index == 1:
        total_sys_mass = np.sum(run_params['true_masses'])
        p = get_orbital_period(total_sys_mass, p)

    mass_error_label = 'Fractional mass error'
    
    if loglog: 
        plt.loglog()
        p = np.log10(p)
        labels = log_param_labels
    else: 
        labels = param_labels
    
    if log_error:
        mass_errors = np.log10(mass_errors)
        filename = f'log_{filename}'
        mass_error_label = 'Log (10) fractional mass error'

    plt.figure(figsize=[15, 9])
    plt.set_cmap('viridis')
    plt.scatter(p, mass_errors, s=150, color='white')
    plt.scatter(p, mass_errors, s=60, c=mass_errors)

    if p_index == 0:
        plt.axvline(run_params['M_maj'], linestyle='--', color='white', label='Mass of major planet')
        plt.legend()
    if p_index == 1:
        plt.axvline(get_orbital_period(total_sys_mass, run_params['a_maj']), linestyle='--', color='white', label='Orbital period of major planet')
        plt.legend()

    ax = plt.gca()
    ax.set_facecolor('xkcd:light grey')

    plt.xlabel(labels[p_index])
    plt.ylabel(mass_error_label)
    plt.title(f'{nameslist} vs Fractional mass error.')
    
    saved_file = plot_path / filename

    plt.savefig(f'{saved_file}.png', dpi=800)

def sensitivity_plot(results, filename, run_params, log_error=True, plot_path=plot_path, loglog=False):
    '''Creates a sensitivity plot for a given set of results.'''
    from scipy.interpolate import LinearNDInterpolator
    from system_generation import relative_orbital_velocity
    title = 'Sensitivity plot for a three-body system with a major planet and a minor planet.'

    varied_param_names = run_params['varied_param_names']
    # select the varied parameters
    p1_index = np.where(np.array(nameslist) == varied_param_names[0])[0].item()
    p2_index = np.where(np.array(nameslist) == varied_param_names[1])[0].item()

    mass_errors = results['mass_errors']
    parameters = results[f'{varied_param_names[0]}, {varied_param_names[1]}']
    p1, p2 = parameters[:, 0], parameters[:, 1]

    if p1_index == 1:
        total_sys_mass = np.sum(results['true_masses'])
        p1 = get_orbital_period(total_sys_mass, p1)
    if p2_index == 1:
        total_sys_mass = np.sum(results['true_masses'])
        p2 = get_orbital_period(total_sys_mass, p2)

    if loglog:
        plt.loglog()
        p1 = np.log10(p1)
        p2 = np.log10(p2)
        labels = log_param_labels
    else:
        labels = param_labels

    # interpolate between the two parameters
    p1_space = np.linspace(np.min(p1), np.max(p2), 500)
    p2_space = np.linspace(np.min(p1), np.max(p2), 500)
    p1_grid, p2_grid = np.meshgrid(p1_space, p2_space)
    interp = LinearNDInterpolator((p1, p2), mass_errors)
    Mass_errors_i = interp(p1_grid, p2_grid)

    cbarlabel = 'Fractional mass error'

    if log_error: # If we want this, change the error to the log of the error.
        Mass_errors_i = np.log10(Mass_errors_i)
        mass_errors = np.log10(mass_errors)
        cbarlabel = 'Log (10) fractional mass error'
        filename = f'log_{filename}'

    plt.figure(figsize=[15, 9])
    plt.set_cmap('viridis')
    plt.pcolormesh(p1_grid, p2_grid, Mass_errors_i, shading='auto')
    plt.scatter(p1, p2, s=150, color='white')
    plt.scatter(p1, p2, s=60, c=mass_errors, label='Uninterpolated data')
    
    # If the parameter we vary is the minor planet mass or semimajor axis, plot the major planet position as reference.
    if p1_index == 0: 
        plt.axvline(run_params['M_maj'], linestyle='--', color='white', label='Mass of major planet')
    if p2_index == 0:
        plt.axhline(run_params['M_maj'], linestyle='--', color='white', label='Mass of major planet')
    if p1_index == 1:
        plt.axvline(get_orbital_period(total_sys_mass, run_params['a_maj']), linestyle='--', color='white', label='Orbital period of major planet')
    if p2_index == 1:
        plt.axhline(get_orbital_period(total_sys_mass, run_params['a_maj']), linestyle='--', color='white', label='Orbital period of major planet')
        
    ax = plt.gca()
    ax.set_facecolor('xkcd:light grey')
    
    plt.xlabel(labels[p1_index])
    plt.ylabel(labels[p2_index])
    plt.title(title)
    plt.legend(loc='lower right')
    plt.colorbar(label=cbarlabel)

    saved_file = plot_path / filename
    
    plt.savefig(f'{saved_file}.png', dpi=800)

def save_run_params_to_file(run_params, output_file):
    import json

    # Convert numpy types to standard python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer): 
            return int(obj)
        elif isinstance(obj, np.floating): 
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj  # Return the object as-is if it's already serializable

    run_params_serializable = {key: convert_to_serializable(value) for key, value in run_params.items()}

    # Save the dictionary as a JSON file
    with open(f'{output_file}.json', 'w') as f:
        json.dump(run_params_serializable, f, indent=4)
    print(f'Run parameters saved to {output_file}.json')

def process_result(path, filename, log_error=True, filter_outliers=False, loglog=True):
    '''Loads results from an h5 file, and then creates an image.'''

    results, run_params = load_result(path, filename, filter_outliers=filter_outliers)
    save_run_params_to_file(run_params, path / filename)
    if len(run_params['varied_param_names']) == 2:
        sensitivity_plot(results, f'{filename}', run_params, log_error, plot_path=path, loglog=loglog)
    else: 
        sensitivity_plot_1param(results, f'{filename}', run_params, log_error, plot_path=path, loglog=loglog)
    
    print(f'file {filename} processed')

