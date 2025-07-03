import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import sys
import os
from pathlib import Path

os.environ["AMUSE_CHANNELS"] = "local"
os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "true"
from amuse.units import units, constants # type: ignore

arbeit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(arbeit_path))
plot_path = arbeit_path / 'Plots'

# set matplotlib parameters
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

font = {'family':'serif',
        'weight':'normal',
        'size'  : 20}

plt.rc('font', **font)

def select_masses(masses, losses, lowest_loss = True):
    # select the masses from the epoch with the lowest loss value
    avg_loss_per_epoch = losses

    if lowest_loss:
        good_mass_indices = np.array([np.all(np.array(mass_list) > 0) for mass_list in masses])
        valid_indices = np.where(good_mass_indices)[0]

        best_idx = valid_indices[np.argmin(avg_loss_per_epoch[valid_indices])]

        masses = masses[best_idx]  
        return masses, best_idx, avg_loss_per_epoch
    else:
        masses = masses[-1]
        best_idx = len(masses)-1
        return masses, best_idx, avg_loss_per_epoch
    
def calculate_mass_error(new_masses, sys, relative = True, sumup = True):
    if relative and sumup:
        return np.sum(abs(new_masses - sys.mass.value_in(units.Msun))/sys.mass.value_in(units.Msun))/len(sys)
    elif relative and sumup==False:
        return abs(new_masses - sys.mass.value_in(units.Msun))/sys.mass.value_in(units.Msun)
    elif relative == False and sumup == True:
        return abs(new_masses - sys.mass.value_in(units.Msun))
    elif sumup:
        return np.sum(abs(new_masses - sys.mass.value_in(units.Msun)))/len(sys)

def save_results(
            path, filename, masses, true_masses, mass_error, avg_loss_per_epoch, 
            varied_param_names, varied_params, pv_unc = None
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
        if pv_unc is not None:
            exp_group.create_dataset('pos_vel_uncertainty,',
                                data=pv_unc)
        elif len(varied_param_names) == 2:
            print('len varied param names = 2')
            exp_group.create_dataset(f'{varied_param_names[0]}, {varied_param_names[1]}', 
                                 data=parameters)
        elif len(varied_param_names) == 1:
            print('len varied param names is 1')
            exp_group.create_dataset(f'{varied_param_names[0]},',
                                 data=parameters)
        else:
            print('did not save anything')
            
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
        sample = qmc.scale(sample_unscaled, [bounds1[0], bounds2[0]], [bounds1[1], bounds2[1]])
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
                for group_name in input_h5.keys():
                    group_path = f'exp_{file_index}'
                    input_h5.copy(group_name, output_h5, name = group_path)
                    break # make sure only the first one is used.

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

            # Comma identifies the changed parameter name
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
    'Initial guess offset (Msun)', 'Positional uncertainty (AU)',
    'Velocity uncertainty (AU/day)'
    ]
# list of log axis labels
log_param_labels = [
    'Minor planet mass (log10(Msun))', 'Minor planet orbital period (log(days))', 
    'Evolve time (log10(days))', 'Tau (log10(days))', 'Cost function points (log)',
    'Major planet mass (log10(Msun))', 'Major planet orbital period (log(days))', 
    'Initial guess offset (log10(Msun))', 'Positional uncertainty (log(AU))',
    'Velocity uncertainty (log(AU/day))'
    ]
# list of all the options varied_param_names can be.
# A bit cumbersome, but a way to match the names we get out of the file
# to a parameter index.
nameslist = [
    'M_min', 'a_min', 'evolve_time', 'tau', 
    'num_points_considered_in_cost_function',
    'M_maj', 'a_maj', 'init_guess_offset', 
    'p_unc', 'v_unc'
    ]

def get_orbital_period(M, a):
    '''Takes in Mass in solar masses, and semimajor axis in AU'''
    a = (a | units.AU).value_in(units.m)
    M = (M | units.Msun).value_in(units.kg)
    G = constants.G.value_in(units.m**3 * units.kg**-1 * units.s**-2)
    p = (2*np.pi*np.sqrt(a**3 / (G*M)) | units.s).value_in(units.day)
    return p

def rescale_losses(losses): # TODO: finish this function and allow the loss to be plotted alongside errors in 1d plots.
    rescaled_losses = None
    return rescaled_losses

def sensitivity_plot_1param(results, filename, run_params, log_error=True, plot_path=plot_path, loglog=False, loss_plot=False):
    '''Creates a 1D sensitivity plot for a given set of results.'''

    # Find out which parameter was varied
    varied_param_name = run_params['varied_param_names'][0]
    p_index = np.where(np.array(nameslist) == varied_param_name)[0].item()

    # Extract the parameters we need to visualize 
    mass_errors = results['mass_errors']
    avg_losses_per_epoch = results['avg_loss_per_epoch']

    final_losses = avg_losses_per_epoch[:, -1]
    # TODO: Scale the losses to be in the same range as the errors.
    # Overplot these and see if the mass error and the loss are actually the same!!!
    
    true_masses = np.sum(results['true_masses'], axis=1)
    true_masses2 = np.sum(results['true_masses'][:2], axis=1)[0]

    M_maj, P_maj = run_params['M_maj'], get_orbital_period(true_masses2, run_params['a_maj'])
    p = results[f'{varied_param_name},']

    if p_index == 1:
        p = get_orbital_period(true_masses, p)

    mass_error_label = 'Fractional mass error'
    
    if loglog: 
        p = np.log10(p)
        M_maj = np.log10(M_maj)
        P_maj = np.log10(P_maj)
        labels = log_param_labels
    else: 
        labels = param_labels
    
    if log_error:
        mass_errors = np.log10(mass_errors)
        filename = f'log_{filename}'
        mass_error_label = 'Log (10) fractional mass error'

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=True, layout='compressed')
    plt.set_cmap('viridis')

    vmin = np.min(mass_errors)
    if log_error:
        vmax = min(np.max(mass_errors), 5)
    vmax = min(np.max(mass_errors), 1e5)

    norm = Normalize(vmin=vmin, vmax=vmax)

    for i, ax in enumerate(axes):
        ax.scatter(p, mass_errors[:, i], s=150, color='white')
        sc = ax.scatter(p, mass_errors[:, i], s=60, c=mass_errors[:, i], norm=norm)

        if p_index == 0:
            ax.axvline(M_maj, linestyle='--', color='white', label='M_maj')
        if p_index == 1:
            ax.axvline(P_maj, linestyle='--', color='white', label='P_maj')

        ax.set_facecolor('xkcd:light grey')
        ax.set_xlabel(labels[p_index])
        ax.grid()
        ax.set_title(f'Relative mass error for body {i+1}')

    axes[0].set_ylabel(mass_error_label)

    saved_file = plot_path / filename
    # fig.tight_layout()
    fig.savefig(f'{saved_file}.png', dpi=800)
    plt.close(fig)

def sensitivity_plot_uncertainty(results, filename, run_params, log_error=True, plot_path=plot_path, loglog=False):
    from scipy.interpolate import LinearNDInterpolator


    # Extract the parameters we need to visualize 
    mass_errors = results['mass_errors']
    avg_losses_per_epoch = results['avg_loss_per_epoch']

    p_v_unc = results['pos_vel_uncertainty,']
    p_unc, v_unc = p_v_unc[:, 0], p_v_unc[:, 1]
    p_unc_index, v_unc_index = 8, 9
    mass_error_label = 'Fractional mass error'

    if loglog:
        p_unc = np.log10(p_unc)
        v_unc = np.log10(v_unc)
        labels = log_param_labels
    else:
        labels = param_labels

    if log_error:
        mass_errors = np.log10(mass_errors)
        filename = f'log_{filename}'
        mass_error_label = 'Log (10) fractional mass error'
    
    vmin = np.min(mass_errors)
    if log_error:
        vmax = min(np.max(mass_errors), 5)
    vmax = min(np.max(mass_errors), 1e5)

    norm = Normalize(vmin=vmin, vmax=vmax)

    # interpolate between the two parameters
    p_unc_space = np.linspace(np.min(p_unc), np.max(p_unc), 500)
    v_unc_space = np.linspace(np.min(v_unc), np.max(v_unc), 500)
    p1_grid, p2_grid = np.meshgrid(p_unc_space, v_unc_space)
    interp = LinearNDInterpolator((p_unc, v_unc), mass_errors)
    Mass_errors_i = interp(p1_grid, p2_grid)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharex=True, sharey=True, layout='compressed')
    plt.set_cmap('viridis')

    threshold = -2 if log_error else 1e-2

    for i, ax in enumerate(axes):
        interp = LinearNDInterpolator((p_unc, v_unc), mass_errors[:, i])
        mass_errors_i = interp(p1_grid, p2_grid)

        pcm = ax.pcolormesh(p1_grid, p2_grid, mass_errors_i, shading='auto', norm=norm)
        ax.scatter(p_unc, v_unc, s=150, color='white')
        sc = ax.scatter(p_unc, v_unc, s=60, c=mass_errors[:, i], norm=norm)
        
        try:
            cs = ax.contour(
                p1_grid, p2_grid, mass_errors_i,
                levels=[threshold],
                colors='black',
                linewidths=1.2,
            )
            ax.clabel(cs, inline=True, fontsize=9,
                      fmt=lambda x: "1e-2" if not log_error else "log10=–2")
        except Exception as e:
            print(f"Contour plotting failed for body {i+1}: {e}")
        
        fig.colorbar(pcm, ax=ax, label=mass_error_label, shrink=0.9, use_gridspec=True)
        ax.set_facecolor('xkcd:light grey')
        ax.set_xlabel(labels[p_unc_index])
        ax.grid()
        ax.set_title(f'Relative mass error for body {i+1}')

    axes[0].set_ylabel(labels[v_unc_index])
    

    saved_file = plot_path / filename
    # fig.tight_layout()
    fig.savefig(f'{saved_file}.png', dpi=800)
    plt.close(fig)


def sensitivity_plot(results, filename, run_params, log_error=True, plot_path=plot_path, loglog=False):
    '''Creates a sensitivity plot for a given set of results.'''
    from scipy.interpolate import LinearNDInterpolator

    varied_param_names = run_params['varied_param_names']
    evolve_time = run_params['evolve_time']
    # select the varied parameters
    p1_index = np.where(np.array(nameslist) == varied_param_names[0])[0].item()
    p2_index = np.where(np.array(nameslist) == varied_param_names[1])[0].item()

    mass_errors = results['mass_errors']
    # Sum the total mass to calculate the orbital period.
    # One with, and one without the outer planet.
    true_masses = np.sum(results['true_masses'], axis=1)
    true_masses2 = np.sum(results['true_masses'][:2], axis=1)[0]

    parameters = results[f'{varied_param_names[0]}, {varied_param_names[1]}']
    p1, p2 = parameters[:, 0], parameters[:, 1]

    if p1_index == 1:
        p1 = get_orbital_period(true_masses, p1)
    if p2_index == 1:
        p2 = get_orbital_period(true_masses, p2)

    M_maj, P_maj = run_params['M_maj'], get_orbital_period(true_masses2, run_params['a_maj'])
    
    if loglog:
        p1 = np.log10(p1)
        p2 = np.log10(p2)
        M_maj = np.log10(M_maj)
        P_maj = np.log10(P_maj)
        evolve_time = np.log10(evolve_time)
        labels = log_param_labels
    else:
        labels = param_labels

    
    # interpolate between the two parameters
    p1_space = np.linspace(np.min(p1), np.max(p1), 500)
    p2_space = np.linspace(np.min(p2), np.max(p2), 500)
    p1_grid, p2_grid = np.meshgrid(p1_space, p2_space)
    interp = LinearNDInterpolator((p1, p2), mass_errors)
    Mass_errors_i = interp(p1_grid, p2_grid)

    cbarlabel = 'Fractional mass error'

    if log_error: # If we want this, change the error to the log of the error.
        Mass_errors_i = np.log10(Mass_errors_i)
        mass_errors = np.log10(mass_errors)
        cbarlabel = 'Log (10) fractional mass error'
        filename = f'log_{filename}'

    vmin = np.min(mass_errors)
    if log_error:
        vmax = min(np.max(mass_errors), 5)
    vmax = min(np.max(mass_errors), 1e5)

    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharex=True, sharey=True, layout='compressed')
    plt.set_cmap('viridis')

    threshold = -2 if log_error else 1e-2

    for i, ax in enumerate(axes):
        interp = LinearNDInterpolator((p1, p2), mass_errors[:, i])
        mass_errors_i = interp(p1_grid, p2_grid)

        pcm = ax.pcolormesh(p1_grid, p2_grid, mass_errors_i, shading='auto', norm=norm)
        ax.scatter(p1, p2, s=150, color='white')
        sc = ax.scatter(p1, p2, s=60, c=mass_errors[:, i], norm=norm)

        try:
            cs = ax.contour(
                p1_grid, p2_grid, mass_errors_i,
                levels=[threshold],
                colors='black',
                linewidths=1.2,
            )
            ax.clabel(cs, inline=True, fontsize=9,
                      fmt=lambda x: "1e-2" if not log_error else "log10=–2")
        except Exception as e:
            print(f"Contour plotting failed for body {i+1}: {e}")
        
        if p1_index == 0 and p2_index != 5:
            ax.axvline(M_maj, linestyle='--', color='white', label='M_maj')
        if p2_index == 0 and p1_index != 5:
            ax.axhline(M_maj, linestyle='--', color='white', label='M_maj')
        if p1_index == 1 and p2_index != 6:
            if p1_index != 2 and p2_index != 2:
                ax.axvline(evolve_time, linestyle='--', color='black', label='Evolve time')
            ax.axvline(P_maj, linestyle='--', color='white', label='P_maj')
        if p2_index == 1 and p1_index != 6:
            if p1_index != 2 and p2_index != 2:
                ax.axhline(evolve_time, linestyle='--', color='black', label='Evolve time')
            ax.axhline(P_maj, linestyle='--', color='white', label='P_maj')

        ax.set_facecolor('xkcd:light grey')
        ax.set_xlabel(labels[p1_index])
        ax.grid()
        ax.set_title(f'Relative mass error for body {i+1}')

    axes[0].set_ylabel(labels[p2_index])
    fig.colorbar(pcm, ax=axes, label=cbarlabel, shrink=1, use_gridspec=True)

    saved_file = plot_path / filename
    # fig.tight_layout()
    fig.savefig(f'{saved_file}.png', dpi=800)
    plt.close(fig)

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
        if (run_params['varied_param_names'] == ['p_unc', 'v_unc']).all():
            sensitivity_plot_uncertainty(results, f'{filename}', run_params, log_error, plot_path=path, loglog=loglog)
        else:
            sensitivity_plot(results, f'{filename}', run_params, log_error, plot_path=path, loglog=loglog)
    elif len(run_params['varied_param_names']) == 1: 
        sensitivity_plot_1param(results, f'{filename}', run_params, log_error, plot_path=path, loglog=loglog)
    print(f'file {filename} processed')

