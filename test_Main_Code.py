
import tensorflow as tf
import Learning.Training_loops as node
import keras
from Learning.BT_optimizer import BachelorThesisOptimizer_with_schedule, BachelorThesisOptimizer, BachelorThesisOptimizer_with_schedule_and_noise, BachelorThesisOptimizerWithRelu
import Learning.Body_info_class as clas


@tf.function
def mse_loss(y_true, y_pred, z):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def non_linearity(x):
    return x


def inverse_of_non_linearity(x):
    return x


def plotGraph():
    import NormalCode.MainCode as mc
    import Data.Horizons as horizons
    import numpy as np
    import matplotlib.pyplot as plt

    def calculate_error(simulated_r, simulated_v, real_r, real_v):
        # Calculate the error as the Euclidean distance between simulated and real positions and velocities
        position_error = np.linalg.norm(
            np.array(simulated_r) - np.array(real_r))
        velocity_error = np.linalg.norm(
            np.array(simulated_v) - np.array(real_v))
        return position_error, velocity_error

    massdict = horizons.get_mass_dict()

    massarray = [[body, massdict[body]] for body in massdict]

    # Sort massarray by mass in descending order
    massarray = sorted(massarray, key=lambda x: x[1], reverse=True)

    bodies = [massarray[i][0] for i in range(len(massarray))]

    # We'll get the first 8 most massive bodies
    first_8_bodies = bodies[:8]

    # Get indices of these bodies
    indices_of_bodies = {body: i for i, body in enumerate(bodies)}

    start_date = '2024-02-28'
    end_date = '2024-09-29'

    # Get the real positions and velocities
    r, v = horizons.get_positions_and_velocities_for_given_dates(
        start_date=start_date, end_date=end_date, bodies=bodies)

    num_bodies_array = []
    error_dict = {body: [] for body in first_8_bodies}

    # Loop through different numbers of most massive bodies
    max_num_bodies = len(massarray)
    for num_bodies in range(3, max_num_bodies + 1):
        print(f"Simulating with {num_bodies} bodies...")
        import Arbeit.Learning.Training_loops as node
        simulated_r, simulated_v, _ = node.initValues_synthetic(
            start_date=start_date,
            end_date=end_date,
            num_of_points_considered_in_cost_function=1,
            tau=0.1,
            bodies=bodies[:num_bodies]
        )

        # Calculate the error compared to the real final positions and velocities for each of the first 8 bodies
        for body in first_8_bodies:
            index = indices_of_bodies[body]
            if index >= num_bodies:
                # The body is not included in the simulation with num_bodies bodies
                # Append None or zero to keep the array length consistent
                error_dict[body].append(None)
                continue

            position_error, velocity_error = calculate_error(
                simulated_r[1][index], simulated_v[1][index], r[1][index], v[1][index])

            error_dict[body].append(position_error)
        num_bodies_array.append(num_bodies)

    # Now plot the errors for each body
    import math

    num_plots = len(first_8_bodies)
    cols = 2  # Number of columns in the subplot grid
    rows = math.ceil(num_plots / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))

    for i, body in enumerate(first_8_bodies):
        row = i // cols
        col = i % cols
        if rows > 1:
            ax = axs[row, col]
        else:
            ax = axs[col]
        # Filter out None values
        body_errors = [
            err if err is not None else np.nan for err in error_dict[body]]
        ax.semilogy(num_bodies_array, body_errors, marker='o')
        ax.set_title(f'{body}')
        ax.set_xlabel('num bodies')
        ax.set_ylabel('Position Error')
        ax.grid(True)

    # Hide any unused subplots
    total_subplots = rows * cols
    if total_subplots > num_plots:
        for i in range(num_plots, total_subplots):
            row = i // cols
            col = i % cols
            if rows > 1:
                fig.delaxes(axs[row, col])
            else:
                fig.delaxes(axs[col])

    plt.tight_layout()
    plt.show()


def plotLossFunction():
    import NormalCode.Fast_Main_Code_Without_storing.fastMainCode as mcFast
    import Data.Horizons as horizons
    import numpy as np
    import matplotlib.pyplot as plt
    import copy

    # Get mass dictionary from Horizons data
    massdict = horizons.get_mass_dict()

    # Create an array of [body, mass] pairs and sort by mass in descending order
    massarray = sorted([[body, massdict[body]]
                        for body in massdict], key=lambda x: x[1], reverse=True)

    bodies = [massarray[i][0] for i in range(len(massarray))]

    # Select the first 2 most massive bodies
    num_bodies = 2
    selected_bodies = bodies[:num_bodies]

    # Define the start and end dates
    start_date = '2024-02-28'
    end_date = '2024-02-29'

    # Get the real positions and velocities
    r, v = horizons.get_positions_and_velocities_for_given_dates(
        start_date=start_date, end_date=end_date, bodies=selected_bodies)

    # Convert positions and velocities to NumPy arrays with dtype float64
    r = np.array(r, dtype=np.float64)
    v = np.array(v, dtype=np.float64)

    # Simulate with correct m
    tau = np.float64(1)  # Time step
    num_steps_in_do_step = 1000  # Number of steps to integrate

    # Original masses
    m_original = np.array([massarray[i][1]
                           for i in range(num_bodies)], dtype=np.float64)

    # Perform numerical integration to get the optimal
    r1 = np.copy(r[0])
    v1 = np.copy(v[0])
    for _ in range(num_steps_in_do_step):
        r1, v1 = mcFast.do_step(
            tau, num_bodies, m_original, r1, v1)
    r[1] = np.copy(r1)
    v[1] = np.copy(v1)
    # Define the range of masses around the original masses
    mass_range = []
    num_points = 1000  # Number of points in the mass range

    # +/- 10 around each mass
    for m in m_original:
        mass_range.append(np.linspace(
            m - 1e1, m + 1e1, num_points, dtype=np.float64))

    # Create a meshgrid of mass values
    M1, M2 = np.meshgrid(mass_range[0], mass_range[1])

    # Ensure meshgrid arrays are of type float64
    M1 = M1.astype(np.float64)
    M2 = M2.astype(np.float64)

    # Initialize an array to store total costs
    total_cost_array = np.zeros_like(M1, dtype=np.float64)

    # Loop over all combinations of masses
    for i in range(M1.shape[0]):
        for j in range(M1.shape[1]):
            m = np.array([M1[i, j], M2[i, j]],
                         dtype=np.float64)  # Current masses

            # Reset positions and velocities as NumPy arrays
            r1 = np.copy(r[0])
            v1 = np.copy(v[0])

            # Perform numerical integration
            for _ in range(num_steps_in_do_step):
                r1, v1 = mcFast.do_step(tau, num_bodies, m, r1, v1)

            # Calculate the squared differences between predicted and true positions and velocities
            position_diff = np.sum(
                (np.array(r1, dtype=np.float64) - r[1]) ** 2)
            velocity_diff = np.sum(
                (np.array(v1, dtype=np.float64) - v[1]) ** 2)

            # Compute total cost (error)
            total_cost = position_diff + velocity_diff

            # Store the total cost
            total_cost_array[i, j] = total_cost

    # Plotting the 3D surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create the surface plot
    surf = ax.plot_surface(M1, M2, total_cost_array,
                           cmap='viridis', edgecolor='none', alpha=0.8)

    # Add labels and title
    ax.set_xlabel(f'Mass of {selected_bodies[0]}')
    ax.set_ylabel(f'Mass of {selected_bodies[1]}')
    ax.set_zlabel('Total Cost (Error)')
    ax.set_title('Error as a Function of Masses')

    # Add a color bar
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Adjust viewing angle
    ax.view_init(elev=30, azim=45)

    plt.show()


def init_optimizer(optimizer, n, lr, use_schedule=False):
    if optimizer == "BT":
        rc = BachelorThesisOptimizer(
            learning_rate=lr, shape=n, convergence_rate=1.0001)
    elif optimizer == "BTWS":
        rc = BachelorThesisOptimizer_with_schedule(
            learning_rate=lr, shape=n, convergence_rate=1.0001, amount_threshold=1e-8)  # Have the optimizer learn one decimal point at a time and after this didn't change for a while, have it fixed and go to the next one
    # elif optimizer == "BTReLU":
    #     rc = BachelorThesisOptimizerWithRelu(
    #         learning_rate=lr, shape=n, convergence_rate=1.0001)
    elif optimizer == "ADAM":
        if use_schedule:
            decay_factor = 0.1  # Decrease by a factor of 10
            decay_steps = 50    # Decrease every 50 epochs

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr,
                decay_steps=decay_steps,
                decay_rate=decay_factor,
                staircase=True
            )
            rc = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            rc = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == "SGDWS":
        decay_factor = 0.1  # Decrease by a factor of 10
        decay_steps = 50  # Decrease every 50 epochs

        # Create the learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=decay_steps,
            decay_rate=decay_factor,
            staircase=True  # Make the decay behave like a step function
        )

        # Create the optimizer with the learning rate schedule
        rc = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    return rc


def test():
    # tf.config.optimizer.set_jit(True)
    import Learning.init_Values as init
    import Data.Horizons as horizons

    # Get mass dictionary from Horizons data
    massdict = horizons.get_mass_dict()

    # Create an array of [body, mass] pairs and sort by mass in descending order
    massarray = sorted([[body, massdict[body]]
                        for body in massdict], key=lambda x: x[1], reverse=True)

    # bodies = [massarray[i][0] for i in range(len(massarray))]

    num_bodies = 3

    massarray = massarray[:num_bodies]

    initial_guess = [20.107, 0.0302, 0.0000201]

    bodies_and_initial_mass_guess_list = [[massarray[i][0], initial_guess[i]]
                                          for i in range(num_bodies)]

    tau = 1
    start_date = '2024-02-28'
    end_date = '2024-09-29'
    unknown_dimension = 3

    optimizer = init_optimizer(
        "BT", num_bodies + num_bodies * 3 * 2, lr=0.1)

    availabe_info_of_bodies = init.initValues(tau=tau, start_date=start_date, end_date=end_date,
                                              num_of_points_considered_in_cost_function=10,
                                              bodies_and_initial_mass_guess_list=bodies_and_initial_mass_guess_list,
                                              synthetic=True, unknown_dimension=unknown_dimension)

    with tf.device('/CPU:0'):
        node.learn_masses(
            tau=tau, optimizer=optimizer,
            availabe_info_of_bodies=availabe_info_of_bodies,
            epochs=100,
            unknown_dimension=unknown_dimension,
            plotGraph=True,
            plot_in_2D=False,
            negative_mass_penalty=100)


# if __name__ == "__main__":

#     plotLossFunction()
