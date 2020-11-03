import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from Neural_Network import Multi_Layer_Neural_Network
from Functions import Eggholder_Function, Eggholder_Function_Limits

# Sets resolution of the function to be approximated
num_grid_points_each_axis = 20

# Sets number iteration the batch training is performed
num_training_iterations = 1000

# Points along each axis
axis = np.linspace(Eggholder_Function_Limits[0], Eggholder_Function_Limits[1], num_grid_points_each_axis)

def target_function(matrix) :

    return Eggholder_Function(matrix[0:1, :], matrix[1:, :])

def f1(x) :

    return np.tanh(x)

def f1_derivative(x) :

    return 1 - np.square(np.tanh(x))

def f2(x) :

    return np.copy(x)

def f2_derivative(x) :

    return np.ones_like(x)

a0 = [[], []]
t  = [[]]

for x in axis :

    for y in axis :
        
       a0[0].append(x)
       a0[1].append(y)
       t[0].append(Eggholder_Function(x,y))

a0 = np.array(a0)
t  = np.array(t)

nn = Multi_Layer_Neural_Network.generate_network_structure(
    [2, np.ceil(0.25 * (num_grid_points_each_axis**2 - 1)), 1],
    [f1, f2],
    [f1_derivative, f2_derivative]
)

nn.set_learning_rate(0.1)
nn.set_learning_rate_decay_factor(0.7)
nn.set_learning_rate_growth_factor(1.05)
nn.set_momentum_coefficient(0.7)
nn.set_critical_error_growth_ratio(0.04)

nn.init_network(train_model=True)

plt.style.use('dark_background')

fig = plt.figure(constrained_layout=True)
gs  = fig.add_gridspec(2,2)

ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])
ax3 = fig.add_subplot(gs[:, 0])

nn.network_graph_init(ax3)
ax3.set_title('Iteration # 0', {'fontsize':15})

c = []
d = []

for x in axis :

    c_buf = []
    d_buf = []

    for y in axis :
        
        c_buf.append(nn.predict(np.array([[x],[y]]))[0,0])
        d_buf.append(target_function(np.array([[x],[y]]))[0,0])

    c.append(c_buf)
    d.append(d_buf)

X, Y = np.meshgrid( axis, axis, indexing='ij'   )

lvl = np.linspace(np.min(d), np.max(d), 20)

cont1 = ax1.contourf(X, Y, np.array(d), cmap='magma', levels=lvl)
ax1.set_xlabel('x')
ax1.set_ylabel('y')

cont2 = ax2.contourf(X, Y, np.array(c), cmap='magma', levels=lvl)
ax2.set_xlabel('x')
ax2.set_ylabel('y')

random_points = Eggholder_Function_Limits[0] + (Eggholder_Function_Limits[1] - Eggholder_Function_Limits[0]) * np.random.rand(2, 40)
np.savetxt('Eggholder Function Approximation Test Points.csv', random_points.transpose(), delimiter=',')

def update(i) :

    global cont2

    nn.train_model(a0, t)
    ax3.set_title(  'Iteration # ' + str(i+1) + '\nMSE = ' + str(round(np.mean(np.square(target_function(random_points) - nn.predict(random_points))), 2)),
                    {'fontsize':15})

    c = []

    for x in axis :

        c_buf = []
    
        for y in axis :
            
            c_buf.append(nn.predict(np.array([[x],[y]]))[0,0])

        c.append(c_buf)

    for objs in cont2.collections : 

        objs.remove()

    cont2 = ax2.contourf(X, Y, np.array(c), cmap='magma', levels=lvl)

    return nn.network_graph_update().append(cont2)

anim = FuncAnimation(fig, update, frames=num_training_iterations, repeat=False)

plt.show()

anim.save('Eggholder Training Model.mp4', writer='ffmpeg', fps=30)