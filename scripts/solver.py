from pyfluids import Fluid, FluidsList, Input
from scipy import constants as cs
import numpy as np
import data.pipe_handler as pp
import data.assignment_data as dh
# import scripts.correlations as rr

def solver() -> tuple:
    # Problem setup
    
    # Disceretize the z domain
    z = np.linspace(-dh.H/2, dh.H/2, dh.n_points)
    
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
    
    # Mass flow rate in the sub-channel
    A_subchannel = dh.pitch**2 - dh.D_fuel_road**2/4 * cs.pi
    mass_flow_subchannel = G_avg * A_subchannel
    
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
        'k':[]
    }
    
    for enth in enthalpy(z):
        if enth < enthalpy_sat_water:
            coolant = water.with_state(Input.enthalpy(enth), Input.pressure(dh.p_coolant))
        else:
            coolant = water_sat
            
        Coolant_profs['T'].append(coolant.temperature)
        Coolant_profs['mu'].append(coolant.dynamic_viscosity)
        Coolant_profs['Pr'].append(coolant.prandtl)
        Coolant_profs['k'].append(coolant.conductivity)
        
    # Equilibrium quality profile
    x_eq = (enthalpy(z)-enthalpy_sat_water)/(enthalpy_sat_steam-enthalpy_sat_water)

    # Outer cladding temperature
    C = 0.042*dh.pitch/dh.D_fuel_road - 0.024
    D_eq = 4*dh.pitch/cs.pi/dh.D_fuel_road-dh.D_fuel_road
    Coolant_profs.update({'Nu':[C*pow((G_avg*D_eq/mu),0.8)*pow((Pr),0.4) for mu,Pr in zip(Coolant_profs['mu'], Coolant_profs['Pr'])]})
    Coolant_profs.update({'h':[Nu*k/D_eq for Nu,k in zip(Coolant_profs['Nu'], Coolant_profs['k'])]})
    
    q2_hot_subchannel = qv(z) * (dh.D_fuel_pellet**2/4) / (dh.D_fuel_pellet)    # W/m2 - total heat in a small cylinder dz / surface area
    T_co_SP = [(T + q/h) for q,T,h in zip(q2_hot_subchannel, Coolant_profs['T'], Coolant_profs['h'])]      # Single phase convection
    T_co_JL = [(T_sat + 25*pow((q*1e-6),0.25)*np.exp(-dh.p_coolant*1e-5/62)) for q in q2_hot_subchannel]   # Jens-Lottes correlation
    
    T_co = [min(SP, JL) for SP,JL in zip(T_co_SP, T_co_JL)]
    
    # Finding the start of the subcooled boiling region
    z_scb = next(z[i] for i in range(len(z)) if T_co_SP[i] > T_co_JL[i])
    
    # Finding the detachment
    #Tl_D = [(T_sat - q/5/h) for q,h in zip(q2_hot_subchannel, Coolant_profs['h'])]
    #z_D = next((z[i] for i in range(len(z)) if Coolant_profs['T'][i] > Tl_D[i]), 0)
    
    #suggerimento gemini per avere la temperatura di distacco 
    z_D_esatta=0
    T_distacco_esatta=0
    # 1. Calcoli l'intero profilo delle temperature (lista)
    Tl_D = [(T_sat - q/5/h) for q,h in zip(q2_hot_subchannel, Coolant_profs['h'])]

    # 2. Trovi l'INDICE del primo punto in cui si verifica il distacco
    # (Se non lo trova, restituisce None per evitare errori)
    indice_distacco = next((i for i in range(len(z)) if Coolant_profs['T'][i] > Tl_D[i]), None)

    # 3. Estrai i due valori singoli che ti interessano
    if indice_distacco is not None:
        z_D_esatta = z[indice_distacco]
        T_distacco_esatta = Tl_D[indice_distacco]
    else:
        # Caso in cui non c'è distacco nel canale
        z_D_esatta = 0
        T_distacco_esatta = None
        
    # FLow quality after the detachment, using the bowring-Rouhani model

    #new working domain 
    
    z_range = z+[0]
    
    #constant values respect to integral, we don't consider pressure drop along z?

    perimeter = cs.pi * dh.D_fuel_road
    mass_flow_subchannel = G_avg * A_subchannel
    H_fg = steam_sat.enthalpy - water_sat.enthalpy
    #enthalpy_sat_water già c'è 
    density_sat_steam = steam_sat.density
    #steam properties

    #T_sat già c'è
    x = 0
    alfa_slip_ratio = []
    for i in range(len(z_range)):
        if z_range[i] >= z_D_esatta:
            dz = - z_range[i] + z_range[i+1]
            eps_Rouhani =  water.with_state(Input.temperature(Coolant_profs['T'][i]), Input.pressure(dh.p_coolant)).density / steam_sat.density/H_fg*(enthalpy_sat_water - enthalpy(z_range[i]))
            q_sp = enthalpy(z_range[i])*mass_flow_subchannel*(T_sat - Coolant_profs['T'][i])
            q2_hot_subchannel_i = qv(z_range[i]) * (dh.D_fuel_pellet**2/4) / (dh.D_fuel_pellet)    # W/m2 - total heat in a small cylinder dz / surface area
            x_i = (q2_hot_subchannel_i-q_sp)/(1+eps_Rouhani)*dz
            x += x_i
            alfa_i = x/(x+(1-x)*pow(density_sat_water/water.with_state(Input.temperature(Coolant_profs['T'][i]), Input.pressure(dh.p_coolant)).density, -2/3))
            alfa_slip_ratio.append(alfa_i)
    flow_quality_Rouhani = x*perimeter/(mass_flow_subchannel*H_fg)

    #void fraction 
    #linear variation of void fraction, maurer correlation to obtain the void fraction at the detachment point
    R_d = 2.37E-3/pow(dh.pressure_coolant, 0.237)
    delta = 0.0666*R_d
    alfa_maurer = 4-delta/D_eq
    #inner cladding temperature 
    #approx the temperature variation as linear.
    k_cl=[qv(z[i])*A_subchannel/(2*CS.pi)*cs.ln(dh.dimeter_fuel_road/(dh.dimeter_fuel_road-2*dh.t_cladding)) for i in range(len(z))]
    T_in_cladding = [T_co[i] + (qv(z[i])*dh.t_cladding**2/K_cl[i]) for i in range(len(z))] #rivedere l'uso del power density
    #fuel pellet surface temperature, dobbiamo iterare ipotesi su temperatura media del fuel e ottenere la temperatura di superficie del fuel
    T_f_avg_guess = 550 #°C
    
    #gap conductance


    return z, [Coolant_profs['T'], T_co], z_scb, [T_co_SP, T_co_JL]
