import numpy as np

# Class representing a layer of a neural network
class Layer :

    # Initialize, create objects under each layer
    def __init__(self) :

        # Number of input to this layer
        self.__num_input = None
        # Number of ouput to this layer
        self.__num_output = None

        # transfer function
        # use single type of transfer function for
        # all neurons in this layer
        self.__transfer_function = None

        # derivative of transfer function
        self.__transfer_function_derivative = None

        # Matrix to hold weights
        # Shape of matrix = number of input x number of output
        # Multiply W.transpose() . input column vector
        self.__W = None

        # Column Vector to hold biases
        # Shape of vector = number of output x 1
        self.__b = None

        # Column Vector to hold inputs
        # to summer during forward propagation
        # a_{m-1}
        self.__a_mm1 = None

        # Column Vector to hold 
        # summer output during forward propagation
        self.__n = None

        # Matrix to hold change in weights
        self.__Delta_W = None
        # Column Vector to hold change in biases
        self.__Delta_b = None

        # Matrix to hold weights in previous stage of learning
        # Shape of matrix = number of input x number of output
        # Multiply W.transpose() . input column vector
        self.__prev_W = None

        # Column Vector to hold biases in previous stage of learning
        # Shape of vector = number of output x 1
        self.__prev_b = None

        # Matrix to hold change in weights in previous stage of learning
        self.__prev_Delta_W = None
        # Column Vector to hold change in biases in previous stage of learning
        self.__prev_Delta_b = None

        # Changeable State Flag
        # If true, only then objects under Layer may be set / reset
        self.__changeable_state = True

        # Learning State Flag
        # If true, only then Layer may be trained
        self.__learning_state = False

        # Forward step taken Flag
        # It true, means a forward step is taken
        # and ready for backpropagation
        self.__forward_step_taken = False

    # Set input dimension
    def set_input_dimension(self, n) :
        ''' Sets dimension of input vector to this layer to n   '''

        if int(n) <= 0 :

            raise ValueError('Given input n =' + str(n) + ' needs to be integer greater than 0!!')

        if self.__changeable_state :
            
            self.__num_input = int(n)

        else :

            raise RuntimeError('Object currently not in a changeable state!!')
        
        pass

    # Set output dimension
    def set_output_dimension(self, n) :
        ''' Sets dimension of output vector to this layer to n   '''

        if int(n) <= 0 :

            raise ValueError('Given input n =' + str(n) + ' needs to be integer greater than 0!!')

        if self.__changeable_state :
            
            self.__num_output = int(n)

        else :

            raise RuntimeError('Object currently not in a changeable state!!')
        
        pass

    # Set transfer function
    def set_transfer_function(self, function) :
        ''' Set the transfer function for each neuron in the layer
            to the input - function (should be callable)    '''

        if not callable(function) :

            raise TypeError('Inpur object function -' + str(function) + ' should be callable!!')

        if self.__changeable_state :

            self.__transfer_function = function

        else :

            raise RuntimeError('Object currently not in a changeable state!!')

        pass

    # Set derivative of transfer function
    def set_derivative_transfer_function(self, function) :
        ''' Set the derivative transfer function for each neuron in the layer
            to the input - function (should be callable)    '''

        if not callable(function) :

            raise TypeError('Inpur object function -' + str(function) + ' should be callable!!')

        if self.__changeable_state :

            self.__transfer_function_derivative = function

        else :

            raise RuntimeError('Object currently not in a changeable state!!')

        pass

    # checks all objects under Layer are defined
    def init_check(self) :

        variables = ''

        if type(self.__num_input) == type(None) :

            variables += 'input dimension, '

        if type(self.__num_output) == type(None) :

            variables += 'output dimension, '

        if type(self.__transfer_function) == type(None) :

            variables += 'transfer function, '

        if len(variables) > 0 :

            raise RuntimeError('Attributes ' + variables[:-2] + ' are not defined yet!!')

        if self.__transfer_function(np.zeros(self.__num_output)).shape[0] != self.__num_output :

            raise ValueError('The transfer function - ' + str(self.__transfer_function) + ' needs to be an ufunc')

        pass        

    # initialise the layer
    def init_layer(self, set_to_learning_mode = False) :
        ''' Allocates space for weights and biases after 
        checking for attributes that need to be predefined '''

        # check all objects under Layer are defined
        self.init_check()

        # Initialize weights matrix with random values
        self.__W = np.random.rand(self.__num_input, self.__num_output)
        # Initialize biases vector with random values
        self.__b = np.random.rand(self.__num_output, 1)

        self.__changeable_state = False

        if set_to_learning_mode :

            self.set_to_training_mode()

        pass

    # run through layer
    def calc_layer(self, input_vector) :
        ''' Returns the output of this layer when input_vector is used -
            f( W.transpose() . x + b )        '''

        if self.__changeable_state :

            raise RuntimeError('First call init_layer!!')

        return self.__transfer_function( np.matmul(self.__W.transpose(), input_vector) + self.__b )

    def get_weights(self) :
        ''' Returns a copy of the current weights matrix    '''

        if self.__changeable_state :

            raise RuntimeError('First call init_layer!!')

        return np.copy(self.__W)

    def get_biases(self) :
        ''' Returns a copy of the current biases vector '''

        if self.__changeable_state :

            raise RuntimeError('First call init_layer!!')

        return np.copy(self.__b)

    def set_weights(self, weight_matrix) :
        ''' Sets the current weights matrix to weight_matrix   '''

        if self.__changeable_state :

            raise RuntimeError('First call init_layer!!')

        if weight_matrix.shape != (self.__num_input, self.__num_output) :

            raise ValueError('Input weight_matrix shape - ' + str(weight_matrix.shape) + ' mismatches with weights ' + str((self.__num_input, self.__num_output)))
        
        np.copyto(self.__W, weight_matrix)
        pass

    def set_biases(self, biases_column_vector) :
        ''' Sets the current biases vector to biases_column_vector    '''

        if self.__changeable_state :

            raise RuntimeError('First call init_layer!!')

        if biases_column_vector.shape != (1, self.__num_output) :

            raise ValueError('Input biases_column_vector shape - ' + str(biases_column_vector.shape) + ' mismatches with biases ' + str((1, self.__num_output)))
        
        np.copyto(self.__b, biases_column_vector)
        pass

    def get_num_neurons(self) :
        ''' Returns the number of neurons in the layer  '''

        if self.__changeable_state :

            raise RuntimeError('First call init_layer!!')

        return self.__num_output

    def get_num_input(self) :
        ''' Returns the number of input to this layer  '''

        if self.__changeable_state :

            raise RuntimeError('First call init_layer!!')

        return self.__num_input

    def set_to_training_mode(self) :
        ''' Checks everything is correctly initialised for training '''

        if self.__changeable_state :

            self.init_layer()

        if type(self.__transfer_function_derivative) == type(None) :

            raise RuntimeError('Derivative of transfer function has not been set!!')

        if self.__transfer_function_derivative(np.zeros(self.__num_output)).shape[0] != self.__num_output :

            raise ValueError('The derivative of transfer function - ' + str(self.__transfer_function_derivative) + ' needs to be an ufunc')

        self.__Delta_W = np.zeros((self.__num_input, self.__num_output), dtype=float)
        self.__Delta_b = np.zeros((self.__num_output, 1), dtype=float)
        self.__prev_W = np.zeros((self.__num_input, self.__num_output), dtype=float)
        self.__prev_b = np.zeros((self.__num_output, 1), dtype=float)
        self.__prev_Delta_W = np.zeros((self.__num_input, self.__num_output), dtype=float)
        self.__prev_Delta_b = np.zeros((self.__num_output, 1), dtype=float)

        self.__learning_state = True
        pass

    def forward_propagation(self, input_vector) :
        ''' Returns tuple of neuron summer values, output of this layer -
            W^T * x + b, f( W^T * x + b )   '''

        if not self.__learning_state :

            raise RuntimeError('First call set_to_training_mode!!')

        if input_vector.shape[0] != self.__num_input :

            raise ValueError('Incorrect size of input vector - ' + str(input_vector.shape) + ', Input size should be - ' + str(self.__num_input))

        if self.__forward_step_taken :

            raise RuntimeWarning('Errors in previous forward propagation haven\'t been backpropagated yet!!')

        self.__a_mm1 = np.copy(input_vector)

        self.__n = np.matmul(self.__W.transpose(), self.__a_mm1) + self.__b

        self.__forward_step_taken = True

        return self.__transfer_function(self.__n)

    def backward_propagation(self, multiplier, alpha = 0.5, gamma = 0.0) :
        ''' Updates weights and biases with learning rate alpha
            and returns the multiplier for the next layer.
            Multiplier here refers to X in s(m) = F'(m)(n(m)) . X
            X is error for m = M and
            X = W(m+1).transpose() . s(m+1) otherwise   '''
            
        if not self.__forward_step_taken :

            raise RuntimeError('Forward Propagation hasn\'t been done yet!!')

        if self.__n.shape != multiplier.shape :

            raise ValueError('Shape of Multiplier incoherent from during forward propagation!! During forward propagation, summer output - ' + str(self.__n.shape) + ', Multiplier - ' + str(multiplier.shape))

        # Sensitivity
        s = np.multiply(multiplier, self.__transfer_function_derivative(self.__n))

        # Save previous step
        np.copyto(self.__prev_Delta_W, self.__Delta_W)
        np.copyto(self.__prev_Delta_b, self.__Delta_b)

        # Change in weights
        np.copyto(
            self.__Delta_W,
            gamma * self.__Delta_W - (1 - gamma) * alpha * np.matmul(self.__a_mm1, s.transpose()) / self.__a_mm1.shape[1]
        )
        # Change in biases
        np.copyto(
            self.__Delta_b,
            gamma * self.__Delta_b - (1 - gamma) * alpha * np.mean(s, axis = 1, keepdims=True)
        )

        # Save previous step before update
        np.copyto(self.__prev_W, self.__W)
        np.copyto(self.__prev_b, self.__b)

        # Weight Update
        self.__W += self.__Delta_W
        # Bias update
        self.__b += self.__Delta_b

        self.__a_mm1 = None
        self.__n = None

        self.__forward_step_taken = False
        # Return multiplier for next layer
        # in backpropagation
        return np.matmul(self.__W, s)

    def reset_to_previous_step(self) :
        ''' Sets the weights and biases to the corresponding values during previous step    '''

        np.copyto(self.__W, self.__prev_W)
        np.copyto(self.__b, self.__prev_b)
        np.copyto(self.__Delta_W, self.__prev_Delta_W)
        np.copyto(self.__Delta_b, self.__prev_Delta_b)
        pass

    @classmethod
    def generate_layer(cls, m, n, f, f_derivative) :
        ''' Returns a predefined, initialised layer with 
            input dimension m, output dimension n, transfer function f,
            and its derivative f_derivative '''

        my_obj = cls()

        my_obj.set_input_dimension(m)
        my_obj.set_output_dimension(n)
        my_obj.set_transfer_function(f)
        my_obj.set_derivative_transfer_function(f_derivative)

        return my_obj


if __name__ == "__main__":

    def target_f(x) :

        return np.sin(x - 0.1)
    
    def f(x) :

        return np.tanh(x)

    def f_derivative(x) :

        return 1 - np.square(np.tanh(x))

    layer = Layer.generate_layer(1, 1, f, f_derivative)

    layer.init_layer(set_to_learning_mode=False)
    layer.set_to_training_mode()

    num = 1000

    a0 = - np.pi / 2 + np.pi * np.random.rand(1, num)
    
    t = target_f(a0)

    # print('Layer Input\n', a0)
    # print('\nTarget\n', t)
   
    
    # print('\nLayer Weights\n', layer.get_weights()), print('\nLayer Biases\n', layer.get_biases())

    for i in range(10) :

        a1 = layer.forward_propagation(a0)
        # if i==0 : print('\nInitial Layer Output\n', a1)
        layer.backward_propagation(- 2 * (t - a1), 0.9, 0.1)

    # print('\nLayer Weights\n', layer.get_weights()), print('\nLayer Biases\n', layer.get_biases())

    p = - np.pi / 2 + np.pi * np.random.rand(1, 1000)
    p = np.sort(p)
    # print('\nFinal Output\n', a1)
    # print('\nTarget\n', t)

    # layer.set_weights(np.ones((1, 1))*10000)
    # layer.set_biases(np.ones((1,1))*0.5*10000)
    print(layer.get_weights())
    print(layer.get_biases())

    a1 = layer.forward_propagation(a0)
    # if i==0 : print('\nInitial Layer Output\n', a1)
    layer.backward_propagation(- 2 * (t - a1), 0.9, 0.1)

    print(layer.get_weights())
    print(layer.get_biases())

    layer.reset_to_previous_step()

    print(layer.get_weights())
    print(layer.get_biases())

    test = np.array(np.isclose(target_f(p), layer.calc_layer(p), atol=0.05), dtype=float)
    print('Probability : ', np.mean(test))

    import matplotlib.pyplot as plt

    plt.plot(p.flatten(), target_f(p).flatten(), label='Target')
    plt.plot(p.flatten(), layer.calc_layer(p).flatten(), label='NN Layer Output')

    plt.legend()
    plt.show()