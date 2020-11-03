import numpy as np


Eggholder_Function_Limits = np.array([-512, 512])

def Eggholder_Function(x, y) :

    return (    - (y + 47) * np.sin( np.sqrt( np.abs( x / 2 + y + 47 ) ) )
                - x * np.sin( np.sqrt( np.abs( x - ( y + 47 ) ) ) )   )



Styblinski_Tang_Function_Limits = np.array([-5, 5])

def Styblinski_Tang_Function(x, y) :
    
    return 0.5 * (x**4 + y**4) - 8 * (x**2 + y**2) + 2.5 * (x + y)
    


Booth_Function_Limits = np.array([-10, 10]) 

def Booth_Function(x, y) :
    
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2
        


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plt.style.use('dark_background')
    plt.rcParams.update({'axes.labelsize':15, 'axes.titlesize':15})
    
    fig = plt.figure()
    ax1 = fig.add_subplot(311, projection='3d')
    ax2 = fig.add_subplot(312, projection='3d')
    ax3 = fig.add_subplot(313, projection='3d')

    n = 100

    axis = np.linspace(Booth_Function_Limits[0], Booth_Function_Limits[1], n)
    X, Y = np.meshgrid(axis, axis)

    ax1.plot_surface(X, Y, Booth_Function(X, Y), cmap='magma')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xticks(np.linspace(Booth_Function_Limits[0], Booth_Function_Limits[1], 5, dtype=int))
    ax1.set_yticks(np.linspace(Booth_Function_Limits[0], Booth_Function_Limits[1], 5, dtype=int))

    axis = np.linspace(Styblinski_Tang_Function_Limits[0], Styblinski_Tang_Function_Limits[1], n)
    X, Y = np.meshgrid(axis, axis)

    ax2.plot_surface(X, Y, Styblinski_Tang_Function(X, Y), cmap='magma')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

        
    axis = np.linspace(Eggholder_Function_Limits[0], Eggholder_Function_Limits[1], n)
    X, Y = np.meshgrid(axis, axis)

    ax3.plot_surface(X, Y, Eggholder_Function(X, Y), cmap='magma')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

    plt.show()