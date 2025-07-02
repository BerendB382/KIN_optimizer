import tensorflow as tf
import TensorCode.backwards_MainCode as mctb
import TensorCode.vectorized_MainCode as vmctb

tf.print('running the backwards.py file')
@tf.function
def vectorized_Gradient_computation(m, rv, newton_solutions, tau):

    # Here the magic happens and the Jacobians are calculated by tensorflow
    # the tape records all operations. tape.jacobian then tells the tape to calculate the jacobian.
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(rv)
        tape.watch(m)

        r, v = tf.split(rv, num_or_size_splits=2, axis=0)
        r = tf.reshape(r, shape=(-1, 3))
        v = tf.reshape(v, shape=(-1, 3))

        r1, v1 = vmctb.do_step(tau, tf.shape(
            r)[0], m, r, v, newton_solutions)

        result = tf.reshape(tf.concat([r1, v1], axis=0), shape=[-1])

    state_gradient = tape.jacobian(result, rv, experimental_use_pfor=False)

    mass_gradient = tape.jacobian(
        result, m, experimental_use_pfor=False)  # TODO: maybe true

    del tape
    return state_gradient, mass_gradient


@tf.function
def gradient_computation(m, rv, newton_solutions, tau):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(rv)
        tape.watch(m)

        r, v = tf.split(rv, num_or_size_splits=2, axis=0)
        r = tf.reshape(r, shape=(-1, 3))
        v = tf.reshape(v, shape=(-1, 3))

        r1, v1 = mctb.do_step(tau, tf.shape(
            r)[0], m, r, v, newton_solutions)

        result = tf.reshape(tf.concat([r1, v1], axis=0), shape=[-1])

    state_gradient = tape.jacobian(result, rv, experimental_use_pfor=False)

    mass_gradient = tape.jacobian(
        result, m, experimental_use_pfor=False)  # TODO: maybe true

    del tape
    return state_gradient, mass_gradient


@ tf.function
def calculate_gradients_for_psiotion(x, m, tau):

    # Here, the state (aka r, v concatinated hence rv)
    # and the notwon solutions are unpacked
    rv, newton_solutions = x

    def true_fn():
        # Here, a nxn matrix is created as there is only one stored root per two body interaction
        newton_solutions_updated = newton_solutions[:, :, 0]
        state_gradient, mass_gradient = vectorized_Gradient_computation(
            m, rv, newton_solutions_updated, tau)
        return state_gradient, mass_gradient

    def false_fn():  # This is comparatively very slow
        state_gradient, mass_gradient = gradient_computation(
            m, rv, newton_solutions, tau)
        return state_gradient, mass_gradient

    # If in the forward pass, the kepler solver didn't have to recursively subdevide tau,
    # there is a way faster backprop version.
    # This version is called vectorized_Gradient_computation because it vectorizes the kepler solver.
    # If there was a need to subdevide tau, no vectorisation is possible. His is way slower, but
    # fortunately rarely happens
    # If there was no need to subdevide, only one newton solution per kepler solver call was stored.
    # This is checked by checking if the second position is zero.
    # newton_solutions is structured like this:
    # It has shape (n, n, 4)
    # The root/roots of the kepler call to solve the kepler problem for body 2 and body 3 is stored at position:
    # newton_solutions[2][3] ( it could also be the other way around. One triangle of the nxn matrix
    # is empty. I think it is the lower one.)
    state_gradient, mass_gradient = tf.cond(
        tf.reduce_all(tf.equal(newton_solutions[:, :, 1], 0.)),
        true_fn,
        false_fn
    )

    return state_gradient, mass_gradient


@ tf.function
def backwards_map_fn(t, tau, m, newton_solutions):
    # The last element in t is not needed, as it would make no sense to compute
    # the derivative of the final time + tau with respect to the derivative of the final state
    t = t[:-1]

    # Here, t and newton solutions are put together in a tuple so that map_fn
    # calls the function calculate_gradients_for_psiotion for each element of them.
    # not a real example but shows the princaple would be:
    # t=[0, 1, 2]
    # newton_solutions = [4, 5, 6]
    # The function calculate_gradients_for_psiotion would then be called for
    # (0, 4), (1, 5), and (2, 6)
    temp = (t, newton_solutions)

    # Here the gradient calculations can happen in parallel.
    # This is done by --like shown above-- calling calculate_gradients_for_psiotion for every element in
    # (t, newton_solutions)
    grads = tf.map_fn(lambda x: calculate_gradients_for_psiotion(
        x, m, tau), temp, fn_output_signature=(tf.float64, tf.float64), swap_memory=True)
    position_derivatives, mass_derivatives = grads

    return position_derivatives, mass_derivatives
