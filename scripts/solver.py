from pyfluids import Fluid, FluidsList, Input
from scipy import constants as cs
import scipy.integrate as integrate
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
    A_subchannel = dh.pitch**2 - dh.D_fuel_road**2/4 * cs.pi
    mass_flow_subchannel = G_avg * A_subchannel
    perimeter_fuel_road = cs.pi * dh.D_fuel_road
    
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
    C = 0.042*dh.pitch/dh.D_fuel_road - 0.024
    D_eq = 4*dh.pitch**2/cs.pi/dh.D_fuel_road-dh.D_fuel_road
    Coolant_profs.update({'Nu':[C*pow((G_avg*D_eq/mu),0.8)*pow((Pr),0.4) for mu,Pr in zip(Coolant_profs['mu'], Coolant_profs['Pr'])]})
    Coolant_profs.update({'h':[Nu*k/D_eq for Nu,k in zip(Coolant_profs['Nu'], Coolant_profs['k'])]})
    
    q2_hot_subchannel = qv(z_vector)/4 * (dh.D_fuel_pellet**2/dh.D_fuel_road)   # W/m2 - total heat in a small cylinder dz / surface area
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
    
    # Inner cladding temperature 
    q_rad_cladding = [qv(z)*A_subchannel/(2*cs.pi)*np.log(dh.D_fuel_road/(dh.D_fuel_road-2*dh.t_cladding)) for z in z_vector] # W/m
    
    # K_cladding constants
    B = 11.45
    A = 1.425e-2 / 2
    
    # Integral term with outer cladding temp
    kT_co = [A*Tz**2 + B*Tz for Tz in T_co]
    C = [- kTz - qz for kTz,qz in zip(kT_co, q_rad_cladding)]
    
    T_ci = list((-B + np.sqrt(B**2 - 4 * A * Cz))/2/A for Cz in C)

    return z_vector, [Coolant_profs['T'], T_co, T_ci], z_NB, z_D, [T_co_SP, T_co_JL], void_fraction
