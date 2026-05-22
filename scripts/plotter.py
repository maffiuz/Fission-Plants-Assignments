import matplotlib.pyplot as plt
import data.assignment_data as dh
import numpy as np

def plotter(results: tuple):
    z, T_profiles, z_scb, T_co_methods, void_fraction, x = results
    
    plt.figure()
    plt.plot(np.transpose(T_profiles),z,label=['Coolant','Cladding out'])
    plt.legend()    
    plt.grid()
    plt.xlabel('Temperature [°C]')
    plt.ylabel('Axial coordinate [m]')
    plt.yticks([-dh.H/2,-1,0,1,dh.H/2],['-H/2','-1','0','1','H/2'])
    plt.ylim((-dh.H/2,dh.H/2))  
    
    plt.figure()
    plt.gca().set_prop_cycle('color',['black','red', 'lime', 'blue'])
    plt.plot(np.transpose(T_co_methods),z,label=['T$_{co,SP}$','T$_{co,J-L}$'])
    plt.plot(T_profiles[1],z,color='lime', linestyle='--',label='T cladding out')
    plt.legend()    
    plt.grid()
    plt.xlabel('Temperature [°C]')
    plt.ylabel('Axial coordinate [m]')
    plt.yticks([-dh.H/2,-1,0,1,dh.H/2],['-H/2','-1','0','1','H/2'])
    plt.ylim((-dh.H/2,dh.H/2)) 
    
    plt.figure()
    plt.plot(z, void_fraction)
    plt.grid()
    
    plt.figure()
    plt.plot(z, x)
    plt.grid()
      
    plt.show()
        