import numpy as np
from itertools import combinations

# Class for a single Radial Basis layer as hidden layer and
# a linear layer on top
class Radial_Basis_Neural_Network :

    def __init__(self) :

        # number of neurons in the radial basis layer
        self.__num_RBF = None

        # number of input
        self.__num_input = None

        # number of ouputs
        self.__num_output = None

        # stores centres of RBFs
        self.__centres = None

        # Weights matrix for 2nd layer
        self.__W = None

        # Bias column vector for 2nd layer
        self.__b = None

        # standard deviations for RBFs
        self.__std_devs = None

        # Flag holding if object is modifiable or not
        self.__is_modifiable = True

    def set_number_of_RBFs(self, n) :
        ''' Sets the number of Radial Basis Functions to use in the hidden layer    '''

        if self.__is_modifiable :

            m = int(n)

            if m > 0 :

                self.__num_RBF = m

            else :

                raise ValueError('Input n - ' + str(n) + ' has to be an integer greater than 0!!')

        else :

            raise RuntimeError('Cannot make changes now!!')

        pass

    def set_number_of_Inputs(self, n) :
        ''' Sets the number of Inputs to be given to the network    '''

        if self.__is_modifiable :

            m = int(n)

            if m > 0 :

                self.__num_input = m

            else :

                raise ValueError('Input n - ' + str(n) + ' has to be an integer greater than 0!!')

        else :

            raise RuntimeError('Cannot make changes now!!')

        pass

    def set_number_of_Outputs(self, n) :
        ''' Sets the number of Outputs to the network '''

        if self.__is_modifiable :

            m = int(n)

            if m > 0 :

                self.__num_output = m

            else :

                raise ValueError('Input n - ' + str(n) + ' has to be an integer greater than 0!!')

        else :

            raise RuntimeError('Cannot make changes now!!')

        pass

    def init_check(self) :
        ''' Checks all parameters of object are set '''

        variables = ''

        if self.__num_input == None :

            variables += 'number of inputs, '

        if self.__num_RBF == None :

            variables += 'number of RBFs, '

        if self.__num_output == None :

            variables += 'number of outputs, '

        if len(variables) > 0 :

            raise RuntimeError('The parameters - ' + variables[:-2] + ' are not defined yet!!')

        pass

    def init_network(self) :
        ''' Initialises memory for weights and biases   '''

        self.init_check()

        self.__centres = np.zeros((self.__num_input, self.__num_RBF), dtype=float)
        
        self.__std_devs = np.zeros((self.__num_RBF, 1), dtype=float)

        self.__W = np.zeros((self.__num_RBF, self.__num_output), dtype=float)

        self.__b = np.zeros((self.__num_output, 1))

        self.__is_modifiable = False
        pass

    def __set_RBF_layer(self, a, t) :
        ''' Clusters training data and finds the centre and diameter for RBFs   '''

        vectors = np.vstack((a, t))

        cluster_ids = cluster(vectors, self.__num_RBF)

        for i in range(self.__num_RBF) :

            input_vectors = a[:, cluster_ids==i]
            
            self.__centres[:, i] = input_vectors.mean(axis=1)
            
            self.__std_devs[i, :] = np.linalg.norm(input_vectors - self.__centres[:, i:i+1], axis=0).max()

        pass
    
    def __calc_RBF_layer_output(self, a) :

        # a[i][j] - c[i][k]
        i, j = np.meshgrid(np.arange(self.__num_RBF), np.arange(a.shape[1]), indexing='ij')

        return np.exp( - 0.5 * ( np.linalg.norm(a[:, j] - self.__centres[:, i], axis=0) / self.__std_devs )**2 ) / ( self.__std_devs * 2 * np.sqrt(np.pi) )

    def train_model(self, a, t) :
        ''' Train RBF model with input matrix a and target matrix t '''

        self.__set_RBF_layer(a, t)

        phi = self.__calc_RBF_layer_output(a)

        phi_inv = np.linalg.pinv(phi) # np.matmul( np.linalg.inv( np.matmul( phi.transpose(), phi ) ), phi.transpose() )

        np.copyto(self.__W, np.matmul( t, phi_inv ).transpose() )

        np.copyto(self.__b, np.mean( t - np.matmul(self.__W.transpose(), phi), axis=1 ))

        pass

    def predict(self, a) :
        ''' Predict output for input column vector a    '''

        phi = self.__calc_RBF_layer_output(a)

        return np.matmul(self.__W.transpose(), phi) + self.__b
        

def cluster(vectors, num_clusters) :
    ''' Returns array of length equal to length of vectors
        containing cluster id to which corresponding vector belongs
        Hierarchical Clustering
        Enter vectors as columns of a matrix    '''

    num_vectors = vectors.shape[1]

    i, j = np.meshgrid(np.arange(num_vectors), np.arange(num_vectors), indexing='ij')
    
    table = np.linalg.norm(vectors[:, i] - vectors[:, j], axis=0)

    cluster_ids = np.arange(num_vectors)

    while num_vectors > num_clusters :

        tril_rows, tril_cols = np.tril_indices_from(table, k=-1,)
        min_index = np.argmin(table[tril_rows, tril_cols])

        i1 = min(tril_rows[min_index], tril_cols[min_index])
        i2 = max(tril_rows[min_index], tril_cols[min_index])

        cluster_ids[cluster_ids == i2] = i1
        cluster_ids[cluster_ids >  i2] -= 1

        table[i1, :] = np.max(np.vstack((table[i1, :], table[i2, :])), axis=0)
        table[:, i1] = np.max(np.vstack((table[:, i1], table[:, i2])), axis=0)

        table = np.delete(table, i2, 0)
        table = np.delete(table, i2, 1)

        num_vectors -= 1

    return cluster_ids

if __name__ == "__main__" :
    
    num = 1000
    x = np.linspace(-1, 1, num).reshape(1, num)
    y = np.sin(x * np.pi) + 1

    nn = Radial_Basis_Neural_Network()
    nn.set_number_of_Inputs(1)
    nn.set_number_of_Outputs(1)
    nn.set_number_of_RBFs(20)

    nn.init_network()

    nn.train_model(x, y)

    import matplotlib.pyplot as plt
    
    plt.plot(x.flatten(), y.flatten(), color='blue')
    plt.plot(x.flatten(), nn.predict(x).flatten(), color='red')

    plt.show()
        