
import tensorflow as tf
import Learning.Forward as forward
import Learning.plotting as pl
import Learning.Combining_derivatives as cd
import Learning.Backwards as b
import queue
import threading
import numpy as np
import traceback

tf.print('running training loops file')

def learn_masses(tau, optimizer, availabe_info_of_bodies,
                 epochs=1,
                 unknown_dimension=0,
                 negative_mass_penalty=0,
                 accuracy=1e-15, plotGraph=True, zoombox=False, 
                 plot_in_2D=False,
                 printing=False):
    '''
    This function is just a sort of wrapper.
    If no graph is being plotted, then the main thread just runs the TensorFlow code
    But if plotGraph=True, then the main thread does the plotting and a new thread does TensorFlow stuff.
    (It doesn't work with matplotlib if not the main thread does GUI things)
    '''
    
    if plotGraph:
        '''
        There are two options for the graphs.
        There is a 3D graph and there is a 2D graph.
        The 2D graph is automatically chosen if one dimension is unknown.
        The 2D and 3D graph can be used if all dimensions are known.
        The 2D graph would then ignore the third dimension and plot [x, y] of [x, y, z]
        '''
        plot_queue = queue.Queue()
        stop_event = threading.Event()

        tensorFlow_thread = threading.Thread(
            target=learn_masses_4real, args=(tau, optimizer, availabe_info_of_bodies, plot_queue, plotGraph,
                                             epochs,
                                             unknown_dimension,
                                             negative_mass_penalty,
                                             accuracy))
        tensorFlow_thread.start()

        pl.plot_thread(plot_queue, stop_event, plot_in_2D, zoombox,
                       unknown_dimension, availabe_info_of_bodies)

    else:
        mass_values, loss_values = learn_masses_4real(tau, optimizer, availabe_info_of_bodies, None, plotGraph,
                           epochs,
                           unknown_dimension,
                           negative_mass_penalty,
                           accuracy,
                           printing = printing)
        return mass_values, loss_values


def learn_masses_4real(tau, optimizer, availabe_info_of_bodies, plot_queue, plotGraph,
                       epochs,
                       unknown_dimension,
                       negative_mass_penalty,
                       accuracy,
                       printing=True):
    '''
    availabe_info_of_bodies may need some explaination.
    It is a list of CelestialBody objects. The code is in the Body_info_class file
    A celestial body object stores all information available for the body. 
    An example would be:
    CelestialBody(
            name="Earth",
            mass=0.001,
            states=[
                TimeState(time=-1,  position=[
                        0.0, 3.0, 12.004], velocity=[0.0, 0.0, 0.1]),
                TimeState(time=8,  position=[0.0, 2.0, 14.0],
                        velocity=[0.1, 345.0, 9.0])
            ]
        )
    This means, that the initial guess for the mass of Earth is: 0.001
    Also note the -1 in time.
    In general, time is measured from start of the simulation in tau units.
    If the first TimeState has time 0, then this is taken to be the known starting position/velocity
    (Except of course for the dimension specified as unknown)
    If it has -1, then everything is taken as initial guess and all dimensions are learned.
    Even if not all three dimentions are known, the position and velocity arrays should still
    have 3 elements. The value for the dimension that is unknown is ignored.
    '''

    # This is important as without it the gpu is used and I haven't tested it with the gpu.
    with tf.device('/CPU:0'):
        tf.print('starting learn_masses_4real')
        # Here, the starting positions, velocities and initial guesses of the masses
        # are taken from availabe_info_of_bodies
        m = []
        r = []
        v = []
        for body in availabe_info_of_bodies:
            m.append(body.mass)
            r.append(body.states[0].position)
            v.append(body.states[0].velocity)

        # Now, the number of timesteps has to be computed.
        # It would be redundant to give this as an input parameter.
        # Obviously nothing is gained if the derivatives of some point
        # in time after the last known position is calculated
        num_total_steps = None
        for body in availabe_info_of_bodies:
            for state in body.states:
                if num_total_steps is None or state.time > num_total_steps:
                    num_total_steps = state.time

        # Here, the starting positions, velocities and initial guesses of the masses
        # are made TensorFlow variables.
        m = tf.Variable(m, dtype=tf.float64)
        r = tf.Variable(r, dtype=tf.float64)
        v = tf.Variable(v, dtype=tf.float64)

        # For the optimizer to optimize all parameters at the same time,
        # they are concatinated into one variable here.
        # This could be changed. Maybe, it makes more sense to have multible optimizers
        # Or maybe not everything should be learned at the same time.
        # But this is the naive simple implementation.
        param = tf.Variable(tf.reshape(tf.concat(
            [m, tf.reshape(tf.concat([r, v], axis=0), shape=[-1])], axis=0), shape=[-1]))

        # Some more transforming stuff into TensorFlow
        tau = tf.constant(tau, tf.float64)
        n = tf.constant(len(r.numpy()), dtype=tf.int32)

        # Maybe this should be a return value?
        # But maybe also the position and velocity values should.
        mass_values = [0 for _ in range(epochs)]
        loss_values = [0 for _ in range(epochs)]
        stop_epoch = 0

        # This is created to store the last value for m.
        m_i_minus_1 = tf.zeros_like(m)

        for j in range(epochs):
            # If the difference between the last value for m and the current one
            # Falls below the threshold of accuracy, the loop is broken.
            if (tf.less_equal(tf.reduce_sum(tf.abs(m_i_minus_1 - m)), accuracy)):
                print(f'loop broken due to accuracy at epoch {stop_epoch+1}')
                break

            # updating the storage for the last value of m
            m_i_minus_1 = tf.identity(m)

            # Now comes the "forward pass".
            # In here, the whole trajectory is computed and each state for each tau
            # timestep is stored in t.
            # Also the roots for the kepler solver are stored in newton_solutions
            if printing:
                print('Forward pass...')
            t, newton_solutions = forward.forward_numpy(
                np.float64(tau.numpy()),
                np.array(r.numpy(), dtype=np.float64),
                np.array(v.numpy(), dtype=np.float64),
                np.array(m.numpy(), dtype=np.float64),
                num_total_steps, int(n.numpy()))

            # This is for communication between the threads. Here, t is given to the other thread
            # The other thread then uses the positions that are in t
            # to update the graph with the blue trajectory lines
            if plotGraph:
                plot_queue.put((t, j))

            # Here, for every element in t (For every state) The network is called to
            # recompute the following state and then backpropagate.
            # The resulting derivatives for d(following state)/d(state) and d(following state)/d(mass)
            # are then stored in state_derivatives, mass_derivatives
            if printing:
                print('Backwards pass...')
            state_derivatives, mass_derivatives = b.backwards_map_fn(
                t, tau, m, newton_solutions)

            # Now, the jacobians of the loss value with respect to the states, where we know how the values or
            # a subset of the values should be have to be calculated.
            # After that, the jacobians for state, loss and mass are combined with the chain rule
            # The chainrule again is:
            # (f(g(x)))'= f'(g(x)) * g'(x)
            if printing:
                print('Combining derivatives...')
            dL_dm, dL_dz, losses = cd.build_compute_graph_and_combine_derivatives(t,
                                                                          availabe_info_of_bodies,
                                                                          state_derivatives,
                                                                          mass_derivatives,
                                                                          negative_mass_penalty,
                                                                          m, unknown_dimension,
                                                                          n)
            gradient = tf.reshape(
                tf.concat([dL_dm, dL_dz], axis=0), shape=[-1])

            optimizer.apply_gradients([(gradient, param)])

            # param is updated. now m, r, v have to be extracted
            m = tf.Variable(param[:n], dtype=tf.float64)
            r = tf.Variable(tf.reshape(
                param[n:3*n + n], shape=(n, 3)), dtype=tf.float64)
            v = tf.Variable(tf.reshape(
                param[3*n + n:], shape=(n, 3)), dtype=tf.float64)

            mass_values[j] = [m.numpy()[i] for i in range(len(m.numpy()))]

            # Sum over all loss components and timesteps
            avg_loss = tf.reduce_sum(losses)
            loss_values[j] = avg_loss.numpy()

            stop_epoch += 1
            if printing:
                # print(
                # f"Epoch {j+1}/{epochs}, Masses: {m.numpy()}, \nPositions: \n{r.numpy()}")
                tf.print(
                f"Epoch {j+1}/{epochs}, Masses: {m.numpy()}, \nLoss = {loss_values[j]}")
                
    mass_values = mass_values[:stop_epoch]
    loss_values = loss_values[:stop_epoch]
    
    return mass_values, loss_values
