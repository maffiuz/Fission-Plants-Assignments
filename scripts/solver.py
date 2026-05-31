from pyfluids import Fluid, FluidsList, Input
from scipy import constants as cs
import scipy.integrate as integrate
from scipy.optimize import fsolve
import numpy as np
# import data.pipe_handler as pp
import data.assignment_data as dh
# import scripts.correlations as rr

def solver() -> tuple:
    # Problem setup
    
    # Disceretize the z domain
    z_vector = np.linspace(-dh.H/2, dh.H/2, dh.n_points)
    
    # Core extrapolated height
    He = dh.H + 1.42 * dh.lambda_tr + 2 * dh.delta
    
    # ================================================
    # 1- Average volumetric heat generation rate
    # Total fuel volume
    Base_area_fuel_pellet = dh.D_fuel_pellet**2 / 4 * cs.pi
    V_fuel = dh.N_rods * Base_area_fuel_pellet * dh.H
    
    qv_avg = dh.P * dh.heat_in_fuel / V_fuel
    
    # ================================================
    # 2- Maximum volumetric heat generation rate
    qv_max = qv_avg * dh.Fq
    qv = lambda z: qv_max*np.cos(cs.pi*z/He)
    
    # ================================================
    # 3- Average coolant mass velocity
    G_avg = dh.mass_flow_coolant / dh.A_flow_coolant
    
    # ================================================
    # 4- Coolant specific enthalpy and temperature profiles
    # Coolant definition
    water = Fluid(FluidsList.Water)
    coolant_in = water.with_state(Input.temperature(dh.T_in_coolant), Input.pressure(dh.p_coolant))
    enthalpy_in = coolant_in.enthalpy
    
    # Sub-channel properties
    A_subchannel = dh.pitch**2 - dh.D_fuel_rod**2/4 * cs.pi
    mass_flow_subchannel = G_avg * A_subchannel
    perimeter_fuel_road = cs.pi * dh.D_fuel_rod
    
    # Enthalpy profile in the coolant (qv_max in MW)
    enthalpy = lambda z: enthalpy_in + 1.0267*(qv_max * Base_area_fuel_pellet * He)/(mass_flow_subchannel * cs.pi)*\
        (np.sin(cs.pi * z / He) +  np.sin(cs.pi * dh.H/2/He))
    
    # Saturation conditions
    water_sat = water.bubble_point_at_pressure(dh.p_coolant)
    steam_sat = water.dew_point_at_pressure(dh.p_coolant)
    enthalpy_sat_water = water_sat.enthalpy
    enthalpy_sat_steam = steam_sat.enthalpy
    T_sat = water_sat.temperature
    
    # Temperature dependent profiles in the coolant
    Coolant_profs = {
        'T':[],
        'mu':[],
        'Pr':[],
        'k':[],
        'rho':[]
    }
    
    for enth in enthalpy(z_vector):
        # Check for saturation conditions
        if enth < enthalpy_sat_water:
            coolant = water.with_state(Input.enthalpy(enth), Input.pressure(dh.p_coolant))
        else:
            coolant = water_sat
            
        Coolant_profs['T'].append(coolant.temperature)
        Coolant_profs['mu'].append(coolant.dynamic_viscosity)
        Coolant_profs['Pr'].append(coolant.prandtl)
        Coolant_profs['k'].append(coolant.conductivity)
        Coolant_profs['rho'].append(coolant.density)
    
    # ================================================
    # 5- Equilibrium quality profile
    x_eq = (enthalpy(z_vector)-enthalpy_sat_water)/(enthalpy_sat_steam-enthalpy_sat_water)

    # ================================================    
    # 6- Cladding outer wall temperature
    C = 0.042*dh.pitch/dh.D_fuel_rod - 0.024
    D_eq = 4*dh.pitch**2/cs.pi/dh.D_fuel_rod-dh.D_fuel_rod
    Coolant_profs.update({'Nu':[C*pow((G_avg*D_eq/mu),0.8)*pow((Pr),0.4) for mu,Pr in zip(Coolant_profs['mu'], Coolant_profs['Pr'])]})
    Coolant_profs.update({'h':[Nu*k/D_eq for Nu,k in zip(Coolant_profs['Nu'], Coolant_profs['k'])]})
    
    q2_hot_subchannel = qv(z_vector)/4 * (dh.D_fuel_pellet**2/dh.D_fuel_rod)   # W/m2 - total heat in a small cylinder dz / surface area
    T_co_SP = [(T + q/h) for q,T,h in zip(q2_hot_subchannel, Coolant_profs['T'], Coolant_profs['h'])]      # Single phase convection
    T_co_JL = [(T_sat + 25*pow((q*1e-6),0.25)*np.exp(-dh.p_coolant*1e-5/62)) for q in q2_hot_subchannel]   # Jens-Lottes correlation
    
    T_co = [min(SP, JL) for SP,JL in zip(T_co_SP, T_co_JL)]
    
    # Finding the start of the subcooled boiling region
    z_NB = next(z_vector[i] for i in range(len(z_vector)) if T_co_SP[i] > T_co_JL[i])
    
    # Finding the detachment
    Tl_D = [(T_sat - q/5/h) for q,h in zip(q2_hot_subchannel, Coolant_profs['h'])]
    z_D = next((z_vector[i] for i in range(len(z_vector)) if Coolant_profs['T'][i] > Tl_D[i]), 0)
        
    # Flow quality after the detachment and void fraction
    # Latent evaporation enthalpy
    H_fg = enthalpy_sat_steam - enthalpy_sat_water

    flow_quality = np.zeros(len(z_vector))
    void_fraction = np.zeros(len(z_vector))
    
    # Integrand function for flow quality
    eps_Rouhani = lambda z: Coolant_profs['rho'][np.searchsorted(z_vector,z)]/steam_sat.density/H_fg*(enthalpy_sat_water - enthalpy(z))
    q_sp = lambda z: Coolant_profs['h'][np.searchsorted(z_vector,z)]*(T_sat - Coolant_profs['T'][np.searchsorted(z_vector,z)])
    q2_hot_subchannel_fun = lambda z: qv(z) * (dh.D_fuel_pellet/4)
    
    integrand = lambda z: perimeter_fuel_road*(q2_hot_subchannel_fun(z) - q_sp(z))/H_fg/mass_flow_subchannel/(1+eps_Rouhani(z))
    
    # Maurer correlation to obtain the void fraction at the detachment point
    R_d = 2.37E-3/pow(dh.p_coolant*1e-5, 0.237)
    delta = 0.0666*R_d
    alpha_Maurer = 4*delta/D_eq
    
    for i in range(len(z_vector)):
        z_actual = z_vector[i]
        
        if z_actual > z_D:    
            z_integr = z_vector[np.searchsorted(z_vector,z_D):i+1]
            
            integrand_vector = []
            for z in z_integr:
                integrand_vector.append(integrand(z))
                
            # Using trapezoidal integration since properties are defined at fixed nodes
            flow_quality[i] = integrate.trapezoid(integrand_vector,z_integr)
            
            # Slip ratio correaltion
            void_fraction[i] = alpha_Maurer + flow_quality[i] / (flow_quality[i] + (1-flow_quality[i]) * pow(Coolant_profs['rho'][i]/steam_sat.density, -2/3))
        
        # Linear variation of void fraction up to detachment
        elif z_actual > z_NB:
            void_fraction[i] = alpha_Maurer * (z_actual - z_NB) / (z_D - z_NB)
    
    # ================================================    
    # 7- Cladding inner wall temperature
    r_in_cladding = (dh.D_fuel_rod-2*dh.t_cladding)/2
    
    # Cladding radial profile temperature is needed to compute the thermal elastic expansion
    r_cladding_vector = np.linspace(r_in_cladding, dh.D_fuel_rod/2,dh.n_points)
    q_rad_cladding_radial = [[qv(z)*Base_area_fuel_pellet/(2*cs.pi)*np.log(dh.D_fuel_rod/2/r) for z in z_vector] 
                      for r in r_cladding_vector] # q'(z,r)
    
    # k_cladding constants
    B = 11.45
    A = 1.425e-2 / 2
    
    # Integral term with outer cladding temp
    kT_co = [A*Tz**2 + B*Tz for Tz in T_co]
    C = [[- kTz - qz for kTz,qz in zip(kT_co, q_rad_cladding)] 
         for q_rad_cladding in q_rad_cladding_radial]
    
    T_c_rad = [[(-B + np.sqrt(B**2 - 4 * A * Cz))/2/A for Cz in Cr] for Cr in C]
    
    T_ci = T_c_rad[0]
    
    # ================================================    
    # 8- Evaluation of the temperature on the surface of the fuel pellet
    # 9- Evaluation of the temperature at the centre of the fuel pellet
    
    # Cladding thickenss after elastic deformation due to pressure difference
    gamma = dh.D_fuel_rod/2/r_in_cladding
    E_zircaloy = lambda T: 1.148e11 - 5.99e7*(T + cs.zero_Celsius)  # [T] in K
    
    E_avg = [(integrate.trapezoid([E_zircaloy(T_c_rad[r][z]) for r in range(len(r_cladding_vector))],r_cladding_vector)/(dh.D_fuel_rod/2 - r_in_cladding)) 
             for z in range(len(z_vector))] # E(z)
    
    Elastic_expansion = [(1/E*(1/(gamma**2-1))*(dh.p_helium_gap*((1-dh.nu_zircaloy)+(1+dh.nu_zircaloy)*gamma**2)-2*gamma**2*dh.p_coolant)) 
                          for E in E_avg]   # Delta r / r (z)
    
    Delta_r_ci_elastic = [r_in_cladding * El for El in Elastic_expansion] # Delta r(z)
    Delta_r_co_elastic = [dh.D_fuel_rod/2 * El for El in Elastic_expansion]
    
    # Cladding thickness after thermal deformation (neglecting elastic deformation)
    T_ambient = 25
    T_c_avg = [(integrate.trapezoid([T_c_rad[r][z] for r in range(len(r_cladding_vector))], r_cladding_vector)/(dh.D_fuel_rod/2 - r_in_cladding)) 
               for z in range(len(z_vector))]
    
    alpha_cladding = [5.62e-6 + 3.162e-9 * T for T in T_c_avg]
    r_in_cladding_th_exp = [r_in_cladding*(1 + alpha*(T - T_ambient)) for alpha,T in zip(alpha_cladding,T_c_avg)]  # r(z)
    r_out_cladding_th_exp = [dh.D_fuel_rod/2*(1 + alpha*(T - T_ambient)) for alpha,T in zip(alpha_cladding,T_c_avg)]  # r(z)
    
    # Total cladding deformation
    r_in_cladding_deformed = [r_in_th + delta_r_in_el for r_in_th,delta_r_in_el in zip(r_in_cladding_th_exp,Delta_r_ci_elastic)] # r(z)
    r_out_cladding_deformed = [r_out_th + delta_r_out_el for r_out_th,delta_r_out_el in zip(r_out_cladding_th_exp,Delta_r_co_elastic)]
    
    cladding_thickness_deformed = [r_out-r_in for r_out,r_in in zip(r_out_cladding_deformed,r_in_cladding_deformed)]
    
    # Fuel thermal expansion properties
    alpha_fuel = lambda T: 7.87e-6 + 3.9e-9 * T
    
    Robertson_factor = 0.96         # -    
    q_rad_fuel = [qv_max * Base_area_fuel_pellet /4/cs.pi * np.cos(cs.pi*z/He) * Robertson_factor for z in z_vector]
    
    # Fuel temperature iterative solution
    toll = 1.e-7
    COUNT = 1000
    i = 0
    T_fs = [T + 50 for T in T_ci]   # °C 
    T_fcl = [500 for _ in z_vector]
    r_out_fuel_th_exp = [dh.D_fuel_pellet/2 for _ in z_vector]
    
    q2_gap = qv(z_vector)/4 * dh.D_fuel_pellet   # W/m2 - total heat in a small cylinder dz / surface area

    # Westinghouse correlation for fuel k
    A = 11.8
    B = 0.0238
    C = 8.775e-13
    
    # Neglecting z-conduction - solving for each height independently
    for i in range(len(z_vector)):
        error_out = 1
        ii = 0
        while error_out > toll and ii < COUNT:
            definite_integral = lambda T: (1/B*np.log(A+B*T) + C/4*T**4)*100
            
            objective_fun = lambda T_cl: definite_integral(T_cl) - definite_integral(T_fs[i]) - q_rad_fuel[i]
            
            T_fcl[i] = fsolve(objective_fun, T_fcl[i])[0]
            
            T_f_avg = (T_fcl[i] + T_fs[i])/2     # Assuming T(r) = f(r^2)
            T_He_gap = (T_fs[i] + T_ci[i])/2
            
            # Fuel thermal expansion
            r_out_fuel_th_exp[i] = dh.D_fuel_pellet/2 * (1 + alpha_fuel(T_f_avg)*(T_f_avg - T_ambient))
            
            # Helium conductivity
            k_He_gap = 0.1763e-2 * pow(T_He_gap, 0.77163)
            delta = r_in_cladding_deformed[i] - r_out_fuel_th_exp[i]
            h_gap = k_He_gap/(2.54e-5 + delta)
            
            # h_gap = k_He_gap/(2.54e-5 + cladding_thickness_deformed[i])
            
            if r_out_fuel_th_exp[i] >= r_in_cladding_deformed[i]:
                print('AIUTO!!!!')
        
            T_fs_new = T_ci[i] + q2_gap[i]/h_gap
            
            error_out = abs(T_fs_new - T_fs[i])/T_fs[i]
            T_fs[i] = T_fs_new
                
            ii += 1
        
    # ================================================        
    # 10 - Critical heat flux
    
    

    return z_vector, [Coolant_profs['T'], T_co, T_ci, T_fs, T_fcl], z_NB, z_D, [T_co_SP, T_co_JL], void_fraction, T_c_rad
