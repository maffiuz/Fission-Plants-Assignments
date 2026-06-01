import matplotlib.pyplot as plt
import data.assignment_data as dh
import numpy as np

def plotter(results: tuple):
    z, T_profiles, z_NB, z_D, T_co_methods, void_fraction, cladding_temp, gap_details, q_C = results
    
    plt.figure()
    plt.title('Temperature profiles')
    plt.plot(np.transpose(T_profiles),z,label=['Coolant','Cladding out','Cladding in','Fuel pellet out','Fuel pellet centerline'])
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
    plt.xlim((260,380))
    
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
    
    plt.figure()
    plt.title('Temperature distribution in the cladding')
    R, Z = np.meshgrid(np.linspace((dh.D_fuel_rod-2*dh.t_cladding)/2,dh.D_fuel_rod/2,dh.n_points),z, indexing='ij')
    plt.pcolormesh(R, Z, cladding_temp, shading='auto', cmap='hot')
    plt.colorbar(label='Temperature [°C]')
    plt.xlabel('Radius')
    plt.ylabel('Axial coordinate [m]')
    plt.yticks([-dh.H/2,-1,0,1,dh.H/2],['-H/2','-1','0','1','H/2'])
    plt.xticks([(dh.D_fuel_rod-2*dh.t_cladding)/2,dh.D_fuel_rod/2],[r'r$_{c,i}$',r'r$_{c,o}$'])
    
    plt.figure()
    plt.title('Thermal deformation of fuel-cladding gap')
    plt.plot(np.transpose(gap_details[0:2]), z, label=['r$_{f,s,th}$','r$_{c,i,th+el}$'])
    plt.axvline(gap_details[2],ls='--',c='black')
    plt.axvline(gap_details[3],ls='--',c='black')
    plt.legend()
    plt.ylabel('Axial coordinate [m]')
    plt.xlabel(r'Radial deformation [m]')
    plt.yticks([-dh.H/2,-1,0,1,dh.H/2],['-H/2','-1','0','1','H/2'])
    plt.ylim((-dh.H/2,dh.H/2)) 
    tick = [gap_details[2] + i/10*(gap_details[3] - gap_details[2]) for i in range(0,11)]
    tick_labl = [r'r$_{f,s}$'] + [f'{i}%' for i in range(10,100,10)] + [r'r$_{c,i}$']
    plt.xticks(tick,tick_labl)
    plt.grid()
    
    plt.figure()
    plt.title('Critical heat flux')
    plt.plot(np.transpose(q_C),z,label=['q$^{''}$ actual','q$_{c,EU}$ W3','q$_{c,NU}$'])
    plt.legend()
    plt.ylabel('Axial coordinate [m]')
    plt.xlabel(r'Radial deformation [m]')
    plt.yticks([-dh.H/2,-1,0,1,dh.H/2],['-H/2','-1','0','1','H/2'])
    plt.grid()
     
    plt.show()
        