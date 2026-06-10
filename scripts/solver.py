from pyfluids import Fluid, FluidsList, Input
from scipy import constants as cs
from scipy.integrate import trapezoid
import numpy as np
import data.assignment_data as dh

def solver():
    # Problem setup
    
    #=================================================
    # 1- Buckling verification
    
    # Loading cladding thermal data
    with open('data/T_cladding_avg','r') as f:
        T_cladding = np.loadtxt(f,skiprows=1)
        
    T_cladding = [T + cs.zero_Celsius for T in np.transpose(T_cladding)]  # °C -> K, first index z
    
    n_points = len(T_cladding[0])
    
    r_cladding_out = dh.D_fuel_rod/2
    r_cladding_in = r_cladding_out - dh.t_cladding
    r_vector = np.linspace(r_cladding_in,r_cladding_out,n_points)
    
    # Evaluating average thermomechanic properties along axial cross-sections
    alpha = []
    E = []
    nu = []

    for i in range(n_points):
        alpha_vec = (6.72e-6*T_cladding[i] - 2.07e-3)/(T_cladding[i] - 308)
        E_vec = (9.9e3 - 5.669*(T_cladding[i] - 273))*9.81
        nu_vec = 0.3303 + 8.376e-5*(T_cladding[i] - 273)
        
        alpha.append(integral_avg(r_vector, alpha_vec))
        E.append(integral_avg(r_vector, E_vec)) 
        nu.append(integral_avg(r_vector, nu_vec))  
        
    # Preliminary buckling pr  
    r_avg = (r_cladding_in + r_cladding_out)/2
    
    p_cr = min(E/(4*(1 - np.pow(nu, 2)))*pow(dh.t_cladding/r_avg,3)) # MPa
    
    #=================================================
    # 2- Maximum internal pressure
    sigma_hoop = dh.sigma_yield
    
    # Mariotte's formule
    p_internal_mariotte = sigma_hoop*dh.t_cladding/r_avg
    
    # Lamé formulation
    K = r_cladding_out/r_cladding_in
    p_internal_lame = sigma_hoop*(K**2 - 1)/(K**2 + 1)
    
    #=================================================
    # 3- Mechanical stresses 
    
    Sm = min(2/3*dh.sigma_yield, 1/3*dh.sigma_ultimate)
      
    p_internal = min(p_cr, p_internal_mariotte, p_internal_lame)
    p_external = dh.p_coolant
    
    A = (p_internal*r_cladding_in**2 - p_external*r_cladding_out**2)/(r_cladding_out**2 - r_cladding_in**2)
    B = ((p_internal - p_external)*r_cladding_in**2*r_cladding_out**2)/(r_cladding_out**2 - r_cladding_in**2)
    
    sigma_hoop_in = A + B/r_cladding_in**2
    sigma_radial_in = A - B/r_cladding_in**2
    sigma_axial_in = sigma_axial_out = (p_internal*r_cladding_in**2 - p_external*r_cladding_out**2)/(r_cladding_out**2 - r_cladding_in**2)
    
    sigma_hoop_out = A + B/r_cladding_out**2
    sigma_radial_out = A - B/r_cladding_out**2
    
    # Primary stress components
    sigma_hoop_avg = (sigma_hoop_in + sigma_hoop_out)/2
    sigma_radial_avg = (sigma_radial_in + sigma_radial_out)/2
    sigma_axial_avg = (sigma_axial_in + sigma_axial_out)/2
    
    #=================================================
    # 4- Thermal stresses

    thermal_exp = []
    DeltaT = []
    sigma_hoop_th_in_vec = []
    sigma_axial_th_in_vec = []
    sigma_hoop_th_out_vec = []
    sigma_axial_th_out_vec = []
    
    for i in range(n_points):
        DeltaT.append(T_cladding[i][0] - T_cladding[i][-1])
        thermal_exp.append(alpha[i]* E[i]/(1-nu[i])*DeltaT[i])
        
        sigma_hoop_th_in_vec.append(-thermal_exp[i]*(K**2/(K**2 - 1) - 1/(2*np.log(K))))
        sigma_axial_th_in_vec.append(sigma_hoop_th_in_vec[i]*(1+nu[i]))
        sigma_hoop_th_out_vec.append(-thermal_exp[i]*(1/(K**2 - 1) - 1/(2*np.log(K))))
        sigma_axial_th_out_vec.append(sigma_hoop_th_out_vec[i]*(1+nu[i]))
        
    sigma_hoop_th_avg = max(sigma_hoop_th_in_vec + sigma_hoop_th_out_vec)/2
    sigma_axial_th_avg = max(sigma_axial_th_in_vec + sigma_axial_th_out_vec)/2
    sigma_radial_th_avg = 0.
    
    sigma_hoop_secondary = sigma_hoop_avg + sigma_hoop_th_avg
    sigma_axial_secondary = sigma_axial_avg + sigma_axial_th_avg
    sigma_radial_secondary = sigma_radial_avg + sigma_radial_th_avg
    
    #=================================================
    # 5- ASME verification
     
    sigma_max_primary = max(abs(sigma_hoop_avg - sigma_radial_avg),
                            abs(sigma_axial_avg - sigma_hoop_avg),
                            abs(sigma_radial_avg - sigma_axial_avg))
    
    print('===========================================')
    print('ASME verification for Primary stress')
    print(f'sigma_max = {sigma_max_primary}')
    print(f'Sm = {Sm}')

    sigma_max_secondary = max(abs(sigma_hoop_secondary - sigma_axial_secondary),
                              abs(sigma_hoop_secondary - sigma_radial_secondary),
                              abs(sigma_radial_secondary - sigma_axial_secondary))
    
    print('===========================================')
    print('ASME verification for Secondary stress')
    print(f'sigma_max = {sigma_max_secondary}')
    print(f'3 Sm = {3*Sm}')
    
    #=================================================
    # 6- Gas plenum sizing
    
    # Total UO2 mass in the fuel rod
    m_UO2 = dh.rho_UO2 * cs.pi * dh.D_fuel_pellet**2/4 * dh.H
    
    # Moles of impurity gases in fuel pellets 
    n_N2 = dh.c_N2*m_UO2/(dh.mm_N2/1000)
    n_H2O = dh.c_H2O*m_UO2/(dh.mm_H2O/1000)
    
    # Moles of gaseous fission products
    mm_U = 238*(1-dh.e_U235) + 235*dh.e_U235
    mm_UO2 = mm_U + 2*16.00
    m_U = m_UO2 * mm_U/mm_UO2
    Ef_MWd = dh.Ef / (60 * 60 * 24) * 1e-6
    NF = dh.BU*m_U/Ef_MWd
    n_Xe_Kr = NF*dh.Y*dh.Rr/cs.Avogadro
    
    # Gas plenum volume and minimum height
    T_plenum = T_cladding[-1][0]
    V_min = (n_H2O + n_N2 + n_Xe_Kr) * cs.R * T_plenum / (p_internal*1e6)
    H_min = V_min/cs.pi/r_cladding_in**2
    H = H_min + 0.15

    print('\n===========================================')
    print(f'Gas plenum total height: {H*100:.2f} cm')
    
    # Checking the ideal gas assumption
    p_H2O = n_H2O*cs.R*T_plenum/V_min
    rho_H2O = n_H2O*dh.mm_H2O*1e-3/V_min
    
    p_H2O_real = Fluid(FluidsList.Water).with_state(Input.temperature(T_plenum),Input.specific_volume(1/rho_H2O)).pressure
    
    err = abs(p_H2O - p_H2O_real)/(p_H2O_real)
    
    print('\n===========================================')
    print('Considering the ideal gas assumption, the partial pressure for H2O was:')
    print(f'p = {p_H2O/1e6:.2f} MPa')
    print(f'real p = {p_H2O_real/1e6:.2f} MPa')
    print(f'relative error = {100*err:.2f} %')


# Handling integral averages over a annulus
def integral_avg(x, y):
    avg = trapezoid(y*x,x)*2/(x[-1]**2 - x[0]**2)
    return avg
