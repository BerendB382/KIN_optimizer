import tensorflow as tf

tf.print('running the compute derivs file')

def calculate_loss_derivatives(availabe_info_of_bodies, t, unknown_dimension, n):

    # There needs to be room for potentially len(t)-1 loss jacobians with each size
    # (1, n * 3 * 2)
    # on index i, dL/dzi+1 is stored
    # The Idea is, that it is thought that every -with the current guess of parameter calculated-
    # state influences the loss by some amount. Even if this amount is zero.
    # In the function scan_fn, it can be seen, that dL_dzfinal += dL_dz.
    # That means, that if dL_dz is zero, nothing happens.
    loss_derivatives = tf.Variable(tf.zeros(
        (tf.shape(t)[0] - 1, 1, n * 3 * 2), dtype=tf.float64))
    losses = tf.Variable(tf.zeros(
        (tf.shape(t)[0] - 1, 1, n * 3 * 2), dtype=tf.float64))

    for i in range(len(availabe_info_of_bodies)):  # for every body

        # for every known state except the startin position.
        # The starting postiion is not influenced by for example wrong guesses of the masses
        for state in availabe_info_of_bodies[i].states[1:]:

            # The jacobian dLoss/d (state that is looked at)
            # It has a weird shape. this is because of the scatter_add function from TensorFlow
            # Just imagine it has shape (1, n * 3 * 2)
            dL_dz = tf.Variable(tf.zeros((1, 1, n * 3 * 2), dtype=tf.float64))
            L = tf.Variable(tf.zeros((1, 1, n * 3 * 2), dtype=tf.float64))
            # In this loop, all availabe information that is known to be true is considered
            for j in range(3):
                # If unknown_dimension=3, then all dimensions are known
                # If not, then all information except this diemstion should be considered.
                if j != unknown_dimension:
                    # The cost function is 1/2 (x-y)**2. The derivative is x-y
                    dL_dz[0, 0, i * 3 + j].assign(t[state.time, i * 3 + j] -
                                                  tf.constant(state.position[j], dtype=tf.float64))
                    dL_dz[0, 0, 3 * n + i * 3 + j].assign(t[state.time, 3 * n + i * 3 + j] - tf.constant(
                        state.velocity[j], dtype=tf.float64))
                    
                    # Also assign loss values
                    L[0, 0, i * 3 + j].assign(0.5*(t[state.time, i * 3 + j] - 
                                                  tf.constant(state.position[j], dtype=tf.float64))**2)
                    L[0, 0, 3 * n + i * 3 + j].assign(0.5*(t[state.time, 3 * n + i * 3 + j] - tf.constant(
                        state.velocity[j], dtype=tf.float64))**2)
            losses = tf.compat.v1.scatter_add(
                losses,
                indices = [state.time - 1],
                updates = L
            )
            loss_derivatives = tf.compat.v1.scatter_add(
                loss_derivatives,
                # The minus one is because the first state is not considered
                indices=[state.time - 1],
                updates=dL_dz
            )

    return loss_derivatives, losses


@tf.function
def combine_derivatives(loss_derivatives, state_derivatives, mass_derivatives, m, negative_mass_penalty, n):
    def scan_fn(state, itfuts):
        dL_dz, dz_dz, dz_dm = itfuts
        dL_dm, dL_dzfinal = state
        dL_dzfinal += dL_dz
        dL_dm = dL_dm + tf.matmul(dL_dzfinal, dz_dm)
        dL_dzfinal = tf.matmul(dL_dzfinal, dz_dz)

        return (dL_dm, dL_dzfinal)
    # Here, the negative mass penalty determines the starting value for the mass derivative.
    # So if a value is negative, -negative_mass_penalty is the derivative, forcing the optimizer
    # to add something to the value.
    dL_dm = tf.where(m >= 0., tf.constant(0.0, dtype=tf.float64),
                     tf.constant(-negative_mass_penalty, dtype=tf.float64))
    dL_dm = tf.expand_dims(dL_dm, 0)

    dL_dz = tf.zeros(shape=(1, n * 3 * 2), dtype=tf.float64)

    dL_dm, dL_dz = tf.scan(
        scan_fn,
        (loss_derivatives, state_derivatives, mass_derivatives),
        initializer=(dL_dm, dL_dz),
        reverse=True
    )

    return tf.reshape(dL_dm[-1], shape=(-1,)), tf.reshape(dL_dz[-1], shape=(-1,))


def set_derivatives_of_known_starting_position_to_zero(availabe_info_of_bodies, unknown_dimension, dL_dz, n):
    dL_dz = tf.Variable(dL_dz)

    # The derivative of where we know the starting position has to be set to zero
    # because we don't want to change that with a gradient descent update
    for i in range(len(availabe_info_of_bodies)):
        if availabe_info_of_bodies[i].states[0].time == 0:
            for j in range(3):
                if j != unknown_dimension:
                    dL_dz[i * 3 + j].assign(tf.constant(0, dtype=tf.float64))
                    dL_dz[3 * n + i * 3 +
                          j].assign(tf.constant(0, dtype=tf.float64))
    return dL_dz


def build_compute_graph_and_combine_derivatives(t, availabe_info_of_bodies,
                                                state_derivatives, mass_derivatives,
                                                negative_mass_penalty, m, unknown_dimension, n):

    loss_derivatives, losses = calculate_loss_derivatives(
        availabe_info_of_bodies, t, unknown_dimension, n)

    dL_dm, dL_dz = combine_derivatives(
        loss_derivatives.read_value(), state_derivatives, mass_derivatives, m, negative_mass_penalty, n)

    dL_dz = set_derivatives_of_known_starting_position_to_zero(
        availabe_info_of_bodies, unknown_dimension, dL_dz, n)
    return dL_dm, dL_dz, losses
