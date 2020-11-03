import numpy as np

import matplotlib.pyplot as plt

from Functions import Eggholder_Function, Eggholder_Function_Limits
from mpl_toolkits.mplot3d import Axes3D
from Radial_Basis_Network import Radial_Basis_Neural_Network

num = 30

axis = np.linspace(Eggholder_Function_Limits[0], Eggholder_Function_Limits[1], num)

a0 = [[], []]
t  = [[]]

for x in axis :

    for y in axis :
        
       a0[0].append(x)
       a0[1].append(y)
       t[0].append(Eggholder_Function(x,y))

a0 = np.array(a0)
t  = np.array(t)

nn = Radial_Basis_Neural_Network()
nn.set_number_of_Inputs(2)
nn.set_number_of_Outputs(1)
nn.set_number_of_RBFs(29)

nn.init_network()

nn.train_model(a0, t)

c = []
d = []

for x in axis :

    c_buf = []
    d_buf = []

    for y in axis :
        
        c_buf.append(nn.predict(np.array([[x],[y]]))[0,0])
        d_buf.append(Eggholder_Function(x, y))

    c.append(c_buf)
    d.append(d_buf)

X, Y = np.meshgrid( axis, axis, indexing='ij'   )

plt.style.use('dark_background')

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

lvl = np.linspace(np.min(d), np.max(d), 20)
print(lvl)
cont1 = ax1.contourf(X, Y, np.array(d), cmap='magma', levels=lvl)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Eggholder Function Surface')

cont2 = ax2.contourf(X, Y, np.array(c), cmap='magma', levels=lvl)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Function Predicted by RBF Network')

random_points = np.genfromtxt('Eggholder Function Approximation Test Points.csv', delimiter=',').transpose()

print(np.round(random_points, 1))

print('\nMSE = ' + str(round(np.mean(np.square(Eggholder_Function(random_points[0, :], random_points[1, :]).reshape(1, -1) - nn.predict(random_points))), 2)))

plt.suptitle('MSE = ' + str(round(np.mean(np.square(Eggholder_Function(random_points[0, :], random_points[1, :]).reshape(1, -1) - nn.predict(random_points))), 2)))
plt.show()
