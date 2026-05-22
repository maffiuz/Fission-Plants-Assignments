from pyfluids import Fluid, FluidsList, Input
from scipy import constants as cs
import scipy.integrate as integrate
import numpy as np
import data.pipe_handler as pp
import data.assignment_data as dh
# import scripts.correlations as rr

def solver() -> tuple:
    # Problem setup
    
    # Disceretize the z domain
    z_vector = np.linspace(-dh.H/2, dh.H/2, dh.n_points)
    dz = z_vector[1]-z_vector[0]
    
    # Core extrapolated height
    He = dh.H + 1.42 * dh.lambda_tr + 2 * dh.delta
    
    # Total fuel volume
    Fuel_pellet_cross_sec = dh.D_fuel_pellet**2 / 4 * cs.pi
    V_fuel = dh.N_rods * Fuel_pellet_cross_sec * dh.H
    
    # Average heat generation rate
    qv_avg = dh.P * dh.heat_in_fuel / V_fuel
    qv_max = qv_avg * dh.Fq
    qv = lambda z: qv_max*np.cos(cs.pi*z/He)
    
    # Average coolant mass velocity
    G_avg = dh.mass_flow_coolant / dh.A_flow_coolant
    
    # Coolant definition
    water = Fluid(FluidsList.Water)
    coolant_in = water.with_state(Input.temperature(dh.T_in_coolant), Input.pressure(dh.p_coolant))
    enthalpy_in = coolant_in.enthalpy
    
    # Sub-channel properties
    A_subchannel = dh.pitch**2 - dh.D_fuel_road**2/4 * cs.pi
    mass_flow_subchannel = G_avg * A_subchannel
    perimeter_fuel_road = cs.pi * dh.D_fuel_road
    
    # Enthalpy profile in the coolant (qv_max in MW)
    enthalpy = lambda z: enthalpy_in + 1.0267*(qv_max * Fuel_pellet_cross_sec * He)/(mass_flow_subchannel * cs.pi)*\
        (np.sin(cs.pi * z / He) +  np.sin(cs.pi * dh.H/2/He))
    
    # Saturation conditions
    water_sat = water.bubble_point_at_pressure(dh.p_coolant)
    steam_sat = water.dew_point_at_pressure(dh.p_coolant)
    enthalpy_sat_water = water_sat.enthalpy
    enthalpy_sat_steam = steam_sat.enthalpy
    T_sat = water_sat.temperature
    
    # Temperature profile in the coolant
    Coolant_profs = {
        'T':[],
        'mu':[],
        'Pr':[],
        'k':[],
        'rho':[]
    }
    
    for enth in enthalpy(z_vector):
        if enth < enthalpy_sat_water:
            coolant = water.with_state(Input.enthalpy(enth), Input.pressure(dh.p_coolant))
        else:
            coolant = water_sat
            
        Coolant_profs['T'].append(coolant.temperature)
        Coolant_profs['mu'].append(coolant.dynamic_viscosity)
        Coolant_profs['Pr'].append(coolant.prandtl)
        Coolant_profs['k'].append(coolant.conductivity)
        Coolant_profs['rho'].append(coolant.density)
        
    # Equilibrium quality profile
    x_eq = (enthalpy(z_vector)-enthalpy_sat_water)/(enthalpy_sat_steam-enthalpy_sat_water)

    # Outer cladding temperature
    C = 0.042*dh.pitch/dh.D_fuel_road - 0.024
    D_eq = 4*dh.pitch**2/cs.pi/dh.D_fuel_road-dh.D_fuel_road
    Coolant_profs.update({'Nu':[C*pow((G_avg*D_eq/mu),0.8)*pow((Pr),0.4) for mu,Pr in zip(Coolant_profs['mu'], Coolant_profs['Pr'])]})
    Coolant_profs.update({'h':[Nu*k/D_eq for Nu,k in zip(Coolant_profs['Nu'], Coolant_profs['k'])]})
    
    q2_hot_subchannel = qv(z_vector) * (dh.D_fuel_pellet/4)   # W/m2 - total heat in a small cylinder dz / surface area
    T_co_SP = [(T + q/h) for q,T,h in zip(q2_hot_subchannel, Coolant_profs['T'], Coolant_profs['h'])]      # Single phase convection
    T_co_JL = [(T_sat + 25*pow((q*1e-6),0.25)*np.exp(-dh.p_coolant*1e-5/62)) for q in q2_hot_subchannel]   # Jens-Lottes correlation
    
    T_co = [min(SP, JL) for SP,JL in zip(T_co_SP, T_co_JL)]
    
    # Finding the start of the subcooled boiling region
    z_NB = next(z_vector[i] for i in range(len(z_vector)) if T_co_SP[i] > T_co_JL[i])
    
    # Finding the detachment
    Tl_D = [(T_sat - q/5/h) for q,h in zip(q2_hot_subchannel, Coolant_profs['h'])]
    z_D = next((z_vector[i] for i in range(len(z_vector)) if Coolant_profs['T'][i] > Tl_D[i]), 0)
        
    # Flow quality after the detachmennt and void fraction
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
                
            # Using trapezoidal integration because properties are defined at fixed nodes
            flow_quality[i] = integrate.trapezoid(integrand_vector,z_integr)
            
            # Slip ratio correaltion
            void_fraction[i] = alpha_Maurer + flow_quality[i] / (flow_quality[i] + (1-flow_quality[i]) * pow(Coolant_profs['rho'][i]/steam_sat.density, -2/3))
        
        # Linear variation of void fraction up to detachment
        elif z_actual > z_NB:
            void_fraction[i] = alpha_Maurer * (z_actual - z_NB) / (z_D - z_NB)
    
    # Inner cladding temperature 
    q_rad_cladding = [qv(z)*A_subchannel/(2*cs.pi)*np.log(dh.D_fuel_road/(dh.D_fuel_road-2*dh.t_cladding)) for z in z_vector] # W/m
    
    # K_cladding constants
    B = 11.45
    A = 1.425e-2 / 2
    
    # Integral term with outer cladding temp
    f_T_CO = A*T_co + B*T_co**2/2
    C = q_rad_cladding + f_T_CO
    
    T_ci = []
    for Cz in C:
        T_ci.append(-B + np.sqrt(B**2 - 4*A*Cz)/2/A)
        
    print(zip(T_co,T_ci))

    #fuel pellet surface temperature, dobbiamo iterare ipotesi su temperatura media del fuel e ottenere la temperatura di superficie del fuel
    T_f_avg_guess = 550 #°C
    
    #gap conductance


    return z_vector, [Coolant_profs['T'], T_co], z_NB, [T_co_SP, T_co_JL], void_fraction, flow_quality, integrand

