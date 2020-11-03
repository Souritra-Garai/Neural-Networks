import numpy as np
from NN_Layer import Layer

# Class representing a multi layer neural network
class Multi_Layer_Neural_Network :

    def __init__(self) :

        # Number of Layers in the network
        self.__num_layers = None
        # List holding Layer type objects 
        # corresponding to each layer in the network
        self.__layers = None

        # State whether neural network is under training
        # If so holds results from intermediate layers in memory
        self.__under_training = False

        # Learning rate
        self.__alpha = None

        # Momentum Coefficient
        self.__gamma = None

        # Growth in learning rate
        self.__eta = None

        # Decay in learning rate
        self.__rho = None

        # Critical ratio for growth in error
        self.__zeta = None

        # holds error at previous stage
        self.__prev_error = None

        # Flag to store whether modifications
        # can be made to the neural network currently
        self.__changeable_state = True

        # List to store matplotlib objects for animation
        self.__matplotlib_objs = None

    # Add a new layer
    def append_layer(self, layer) :
        ''' Append a new layer to the neural network    '''

        if type(layer) != type(Layer()) :

            raise ValueError('layer has to be a Layer() object!!')

        if self.__changeable_state :

            layer.init_layer()

            if type(self.__num_layers) == type(None) :

                self.__layers = [layer]
                self.__num_layers = 1

            else :

                if self.__layers[-1].get_num_neurons() == layer.get_num_input() :

                    self.__layers.append(layer)
                    self.__num_layers += 1

                else :

                    raise ValueError('Given layer has input - ' + str(layer.get_num_input()) + ', that mismatches with previous layer with output - ' + str(self.__layers[-1].get_num_neurons()))

        else :

            raise RuntimeError('Object currently not in a changeable state!!')

        pass

    # Set the learning rate for backpropagation algorithm
    def set_learning_rate(self, alpha) :
        ''' Sets the learning rate for backpropagation algorithm to alpha   '''

        a = float(alpha)

        if a >= 1 and a <= 0 :

            raise ValueError('Learning rate has to be a float number in [0,1] !!')

        self.__alpha = a

        pass

    # Set the momentum coefficient for MOBP
    def set_momentum_coefficient(self, gamma) :
        ''' Sets the momentum coefficient for MOBP to gamma '''

        g = float(gamma)

        if g >= 1 and g <= 0 :

            raise ValueError('Momentum Coefficient has to be a float in [0,1] !!')

        self.__gamma = g

        pass

    def set_critical_error_growth_ratio(self, zeta) :
        ''' Sets the critical ratio for growth in error during VOBP to zeta   '''

        a = float(zeta)

        if a >= 1 and a <= 0 :

            raise ValueError('Critical ratio has to be a float number in [0,1] !!')

        self.__zeta = a

        pass

    def set_learning_rate_growth_factor(self, eta) :
        ''' Sets learning rate growth factor for VOBP to eta '''

        g = float(eta)

        if g <= 1 :

            raise ValueError('Growth factor has to be a float in (1, inf) !!')

        self.__eta = g

        pass

    def set_learning_rate_decay_factor(self, rho) :
        ''' Sets learning rate growth factor for VOBP to eta '''

        g = float(rho)

        if g >= 1 and g <= 0 :

            raise ValueError('Decay factor has to be a float in [0, 1] !!')

        self.__rho = g

        pass

    # Check and initialise the neural network
    def init_network(self, train_model=False) :
        ''' Checks and allocates memory spaces for runtime variables    
            If network needs to be trained, put argument train_model True'''

        for layer in self.__layers :

            layer.init_layer()

        self.__changeable_state = False

        if train_model :

            self.set_to_training_mode()

        pass

    # Sets to training mode
    def set_to_training_mode(self) :
        ''' Sets the neural network to training mode
            Allocates memory for additional variables required for training '''

        if self.__changeable_state :

            raise RuntimeError('Still in a modifiable state!! Call init_network')

        if type(self.__alpha) == type(None) :

            raise RuntimeError('Learning rate alpha is not set!! Call set_learning_rate')

        if type(self.__gamma) == type(None) :

            raise RuntimeError('Momentutm coefficient gamma is not set!! Call set_momentum_coefficient')

        if type(self.__eta) == type(None) :

            raise RuntimeError('Growth factor for learning rate eta is not set!! Call set_learning_rate_growth_factor')

        if type(self.__rho) == type(None) :

            raise RuntimeError('Decay factor for learning rate rho is not set!! Call set_learning_rate_decay_factor')

        if type(self.__zeta) == type(None) :

            raise RuntimeError('Critical ratio for growth in error is not set!! Call set_critical_error_growth_ratio')

        for layer in self.__layers :

            layer.set_to_training_mode()      

        self.__prev_error = np.inf  

        self.__under_training = True
        pass

    # Switch off training mode
    def switch_off_training_mode(self) :
        ''' Switches off training mode
            Stops saving values in intermediate layers '''

        self.__under_training = False
        pass

    # train the model
    def train_model(self, input_vector_matrix, target_vector_matrix) :

        if not self.__under_training :

            raise RuntimeError('Currently not under training!! Call set_to_training_mode')

        if input_vector_matrix.shape[0] != self.__layers[0].get_num_input() :

            raise ValueError('The number of elements in input vector - ' + str(input_vector_matrix.shape[0]) + ' mismatches the number of scalar input set for the model - ' + str(self.__layers[0].get_num_input()))

        if target_vector_matrix.shape[0] != self.__layers[-1].get_num_neurons() :

            raise ValueError('The number of elements in target vector - ' + str(target_vector_matrix.shape[0]) + ' mismatches the number of scalar outputs set for the model - ' + str(self.__layers[-1].get_num_neurons()))

        error = np.mean(np.square(target_vector_matrix - self.predict(input_vector_matrix)))

        gamma = self.__gamma

        if error - self.__prev_error > self.__zeta * self.__prev_error :

            for layer in self.__layers :

                layer.reset_to_previous_step()

            error = self.__prev_error
            self.__alpha *= self.__rho
            gamma = 0.0

        elif error < self.__prev_error :

            self.__alpha *= self.__eta

        self.__prev_error = error        
        
        a = np.copy(input_vector_matrix)

        for layer in self.__layers :

            a = layer.forward_propagation(a)
        
        m = - 2 * (target_vector_matrix - a)

        for layer in reversed(self.__layers) :

            m = layer.backward_propagation(m, self.__alpha, gamma)

        pass

    # Calc Value
    def predict(self, input_vector_matrix) :
        ''' Calculate the output approximated by neural network model
            for the input vector a at layer i   '''

        if self.__changeable_state :

            raise RuntimeError('Still in a modifiable state!! Call init_network')

        if input_vector_matrix.shape[0] != self.__layers[0].get_num_input() :

            raise ValueError('Input shape - ' + str(input_vector_matrix.shape[0]) + ' mismatches with layer input shape - ' + str(self.__layers[0].get_num_input()))
        
        a = np.copy(input_vector_matrix)

        for layer in self.__layers :

            a = layer.calc_layer(a)

        return a

    def network_graph_init(self, axes, node_color='yellow') :
        ''' Initializes matplotlib objects for animation    '''

        layer_coord = np.linspace(0, 1, self.__num_layers + 3)[1:-1]

        self.__matplotlib_objs = []

        for i, layer in enumerate(self.__layers) :

            input_y = np.linspace(-1, 1, layer.get_num_input()+2)[1:-1]
            output_y = np.linspace(-1, 1, layer.get_num_neurons()+2)[1:-1]

            w = np.abs(layer.get_weights())
            w -= w.min()
            w /= w.max()

            layer_objs = []

            for j in range(layer.get_num_input()) :

                layer_input_objs = []

                for k in range(layer.get_num_neurons()) :

                    layer_input_objs.append(axes.plot([layer_coord[i], layer_coord[i+1]], [input_y[j], output_y[k]], c=[w[j][k], w[j][k], w[j][k]], alpha=w[j][k])[0])

                axes.scatter(layer_coord[i], input_y[j], color=node_color)

                layer_objs.append(layer_input_objs)

            self.__matplotlib_objs.append(layer_objs)

            if i == self.__num_layers - 1 :

                for k in range(layer.get_num_neurons()) :

                    axes.scatter(layer_coord[i+1], output_y[k], color=node_color)

        axes.set_xlim([0,1])
        axes.set_ylim([-1, 1])
        axes.axis('off')

        return self.__matplotlib_objs

    def network_graph_update(self) :
        ''' Updates edge colors based on weights    '''

        for i, layer in enumerate(self.__layers) :

            w = np.abs(layer.get_weights())
            w -= w.min()
            w /= w.max()

            for j in range(layer.get_num_input()) :

                for k in range(layer.get_num_neurons()) :

                    self.__matplotlib_objs[i][j][k].set_color([w[j, k]]*4)

        return self.__matplotlib_objs

    @classmethod
    def generate_network_structure(cls, nn_structure, transfer_function_list, transfer_function_derivative_list) :

        my_obj = cls()

        for num_input, num_output, transfer_function, transfer_function_derivative in zip(nn_structure[:-1], nn_structure[1:], transfer_function_list, transfer_function_derivative_list) :

            my_obj.append_layer(Layer.generate_layer(num_input, num_output, transfer_function, transfer_function_derivative))

        return my_obj

if __name__ == "__main__":
    
    def target_function(x) :

        return np.sin(3 * x * np.pi )

    def f1(x) :

        return np.tanh(x)

    def f1_derivative(x) :

        return 1 - np.square(np.tanh(x))

    def f2(x) :

        return np.copy(x)

    def f2_derivative(x) :

        return np.ones_like(x)

    nn = Multi_Layer_Neural_Network.generate_network_structure(
        [1, 20, 1],
        [f1, f2],
        [f1_derivative, f2_derivative]
    )

    nn.set_learning_rate(0.1)
    nn.set_learning_rate_decay_factor(0.7)
    nn.set_learning_rate_growth_factor(1.05)
    nn.set_momentum_coefficient(0.9)
    nn.set_critical_error_growth_ratio(0.04)

    nn.init_network(train_model=True)

    x = np.linspace(-1, 1, 1000).reshape(1, -1)
    y = target_function(x)

    for i in range(10000) :
        
        nn.train_model(x, y)

    import matplotlib.pyplot as plt

    plt.plot(x.flatten(), y.flatten())
    plt.plot(x.flatten(), nn.predict(x).flatten())

    plt.show()

