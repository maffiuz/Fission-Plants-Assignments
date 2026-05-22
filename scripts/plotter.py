import matplotlib.pyplot as plt
import data.assignment_data as dh
import numpy as np

def plotter(results: tuple):
    z, T_profiles, z_NB, z_D, T_co_methods, void_fraction = results
    
    plt.figure()
    plt.title('Temperature profiles')
    plt.plot(np.transpose(T_profiles),z,label=['Coolant','Cladding out','Cladding in'])
    plt.legend()    
    plt.grid()
    plt.xlabel('Temperature [°C]')
    plt.ylabel('Axial coordinate [m]')
    plt.yticks([-dh.H/2,-1,0,1,dh.H/2],['-H/2','-1','0','1','H/2'])
    plt.ylim((-dh.H/2,dh.H/2))  
    
    plt.figure()
    plt.title('Outer cladding temperature analysis')
    plt.gca().set_prop_cycle('color',['black','red', 'lime'])
    plt.plot(np.transpose(T_co_methods),z,label=['T$_{co,SP}$','T$_{co,J-L}$'])
    plt.plot(T_profiles[1],z,color='lime', linestyle='--',label='T cladding out')
    plt.legend()    
    plt.grid()
    plt.xlabel('Temperature [°C]')
    plt.ylabel('Axial coordinate [m]')
    plt.yticks([-dh.H/2,-1,0,1,dh.H/2],['-H/2','-1','0','1','H/2'])
    plt.ylim((-dh.H/2,dh.H/2)) 
    
    plt.figure()
    plt.title('Coolant void fraction due to boiling')
    plt.plot(void_fraction, z)
    plt.axhline(z_NB,ls='--',c='black',label='z$_{NB}$')
    plt.axhline(z_D,ls='-.',c='black',label='z$_{D}$')
    plt.legend()
    plt.ylabel('Axial coordinate [m]')
    plt.xlabel(r'Void fraction $\alpha$ [-]')
    plt.yticks([-dh.H/2,-1,0,1,dh.H/2],['-H/2','-1','0','1','H/2'])
    plt.ylim((-dh.H/2,dh.H/2)) 
    plt.grid()
      
    plt.show()
        