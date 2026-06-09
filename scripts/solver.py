from pyfluids import Fluid, FluidsList, Input
from scipy import constants as cs
import data.pipe_handler as pp
import data.assignment_data as dh
import scripts.correlations as rr
import math as m
import pandas as pd

def solver():
    # Problem setup
    water = Fluid(FluidsList.Water)
    
    # HX2 fixed parameters
    q2_HX2 = dh.Pth/dh.Atransfer_HX2            # Power flux
    Delta_Tsat = rr.inverse_McAdams(q2_HX2)
    h_out_HX2 = q2_HX2/Delta_Tsat
    Din_HX2 = dh.Dout_HX2 - 2*dh.thick_HX2

    # Leg fixed parameters
    _, Din_ISC = pp.get_diameter(pp.get_row(dh.Dout_ISC), col='100')
    
    # HX1 shell fixed parameters 
    A_shell = rr.shell_flow_area(dh.Din_shell_HX1,dh.pitch_HX1,dh.Dout_HX1,dh.lbaffles_HX1)

    # Iterative quantities for outer loop (mass flow rate)
    err_out = 1
    toll_out = 1.e-5
    cc_out = 0
    COUNT = 1e1
    
    # Initial guesses
    mass_flow = 100     # kg/s
    T_hot = 160         # °C
    
    while err_out > toll_out and cc_out < COUNT:
        # print(f'{cc_out} iter: mass_rate = {mass_flow:.4f}, relative err: {err_out:.3e}')
        
        # Iterative quantities for inner loop (Temperature)
        err_in = 1
        toll_in = 1.e-5
        cc_in = 0
        
        # Temperature independent parameters
        Delta_h = dh.Pth/mass_flow
        mass_flow_pipe_HX2 = mass_flow/dh.Ntubes_HX2

        while err_in > toll_in and cc_in < COUNT:
            # print(f'{cc_in} iter: T hot = {T_hot:.4f}, relative err: {err_in:.3e}')
            
            # Fluid states definition
            w_hot = water.with_state(Input.temperature(T_hot), Input.pressure(dh.p_ISC))    # Hot fluid
            h_hot = w_hot.enthalpy
            h_cold = h_hot - Delta_h
            w_cold = water.with_state(Input.pressure(dh.p_ISC), Input.enthalpy(h_cold))     # Cold fluid
            T_cold = w_cold.temperature
            T_avg = (T_hot + T_cold)/2
            w_avg = water.with_state(Input.temperature(T_avg), Input.pressure(dh.p_ISC))    # Average fluid (HX1, HX2)
            
            # Fluid properties in HX2 (T_avg) 
            Nu = rr.Nusselt(w_avg, mass_flow_pipe_HX2, Din_HX2, dh.epsilon_rel_HX2)
            k_w = w_avg.conductivity
            cp_w = w_avg.specific_heat
            h_in_HX2 = Nu * k_w / Din_HX2
            
            # Solving HX2
            inverse_Uw_HX2 = dh.Dout_HX2/Din_HX2/h_in_HX2 + dh.Dout_HX2*m.log(dh.Dout_HX2/Din_HX2)/2/dh.k_HX2 + 1/h_out_HX2
            Uw_HX2 = pow(inverse_Uw_HX2 ,-1)                    # Global heat transfer coeff
            NTUw_HX2 = Uw_HX2*dh.Atransfer_HX2/mass_flow/cp_w   # Number transfer unit
            DeltaT_ML_HX2 = q2_HX2 / Uw_HX2                     # Delta T mean log
            
            # Fining the new T_hot
            T_hot_new = 100 + DeltaT_ML_HX2*NTUw_HX2/(1-m.exp(-NTUw_HX2))
            
            # Updating inner loop parameters
            cc_in += 1
            err_in = abs(T_hot-T_hot_new)/T_hot
            
            T_hot = T_hot_new
        
        # print(f'{cc_in} iter: T hot = {T_hot:.4f}, relative err: {err_in:.3e}')
        
        # Re-definition of fluids for the final T
        w_hot = water.with_state(Input.temperature(T_hot), Input.pressure(dh.p_ISC))
        h_hot = w_hot.enthalpy
        h_cold = h_hot - Delta_h
        w_cold = water.with_state(Input.pressure(dh.p_ISC), Input.enthalpy(h_cold))
        T_cold = w_cold.temperature
        T_avg = (T_hot + T_cold)/2

        # Properties of hot water
        rho_hot = w_hot.density
        Re_hot = rr.Reynolds_massrate(w_hot,mass_flow,Din_ISC)
        # Properties of cold water
        rho_cold = w_cold.density
        Re_cold = rr.Reynolds_massrate(w_cold,mass_flow,Din_ISC)
        
        # Losses of hot leg ISC
        # Localized 
        loc_90_H = rr.localized_loss_coeff(dh.k_loss_ISC,rho_hot,Din_ISC)           # 90° bends
        # Distributed 
        f_hot = rr.friction_rough(Re_hot,dh.epsilon_rel_ISC)
        dist_H_leg = rr.distributed_loss_coeff(dh.L_H_ISC,Din_ISC,f_hot,rho_hot)
        # Total losses inside the hot leg ISC
        Hot_losses = dh.N_bends_ISC * loc_90_H + dist_H_leg
    
        # Losses of cold leg ISC     
        # Localized 
        loc_90_C = rr.localized_loss_coeff(dh.k_loss_ISC,rho_cold,Din_ISC)
        # Distributed 
        f_cold = rr.friction_rough(Re_cold,dh.epsilon_rel_ISC,)
        dist_C_leg = rr.distributed_loss_coeff(dh.L_H_ISC,Din_ISC,f_cold,rho_cold)
        # Total losses inside the cold leg ISC
        Cold_losses = dh.N_bends_ISC * loc_90_C + dist_C_leg
        
        # Losses inside the HX1 shell
        # Fluid parameters whitin shell
        w_ave = water.with_state(Input.temperature(T_avg), Input.pressure(dh.p_ISC))
        rho_ave = w_ave.density
        Re_shell = rr.Reynolds_massrate(w_ave,mass_flow,dh.Din_shell_HX1)
        
        # Localized pressure drop coefficient
        k_shell = rr.local_loss_HX1(Re_shell,dh.Din_shell_HX1,dh.Dout_HX1,dh.pitch_HX1)
        Shell_HX1_losses = 0.5 * k_shell /rho_ave/A_shell**2
        
        # Losses inside the HX2 shell
        k_contraction_inf = 0.5
        Re_HX2 = rr.Reynolds_massrate(w_ave,mass_flow_pipe_HX2,Din_HX2)
        f_HX2 = rr.friction_rough(Re_HX2,dh.epsilon_rel_HX2)
        HX2_distrubed_losses = rr.distributed_loss_coeff(dh.Ltubes_HX2,Din_HX2,f_HX2,rho_ave)/dh.Ntubes_HX2**2
        HX2_loc_losses = rr.localized_loss_coeff(4,rho_ave,Din_HX2)/dh.Ntubes_HX2**2
        HX2_losses = HX2_distrubed_losses + HX2_loc_losses
        # HX2_losses = 0
        
        # Gravitational potential
        Potential = ((rho_cold * (dh.H_ISC + dh.Din_shell_HX1)) - (rho_hot * dh.H_ISC) - (rho_ave * dh.Din_shell_HX1)) * cs.g
        
        # New mass rate
        mass_flow_new = pow(Potential/(Hot_losses + Cold_losses + Shell_HX1_losses + HX2_losses), 0.5)
        
        # Updating outer loop parameters
        cc_out += 1
        err_out = abs(mass_flow-mass_flow_new)/mass_flow
            
        mass_flow = mass_flow_new
    
    print('---------------------------------------------------------')
    print(f'ISC iteration complete:\nmass flow rate = {mass_flow:.4f} kg/s\nAverage temperature = {T_avg:.4f} °C\nHot channel temperature = {T_hot:.4f} °C')
    print('---------------------------------------------------------')
    
    # Definition of definitive ISC quantities
    T_hot_ISC = T_hot
    T_cold_ISC = T_cold
    mass_flow_ISC = mass_flow
    Pr_shell = w_ave.prandtl
    kth_shell = w_ave.conductivity
    
    # Writing all useful data on .csv
    data = {
        'Pressure drop location':[],
        'Pressure drop value [Pa]':[]
    }
    
    # ISC buoyancy
    data['Pressure drop location'].append('ISC - Driving buoyancy')
    data['Pressure drop value [Pa]'].append(Potential)
        
    # ISC total losses
    data['Pressure drop location'].append('ISC - Total pressure drop')
    data['Pressure drop value [Pa]'].append((Hot_losses + Cold_losses + Shell_HX1_losses + HX2_losses)*mass_flow_ISC**2)
    
    # ISC hot leg bends
    data['Pressure drop location'].append('ISC - Hot leg 90 deg bends (per bend)')
    data['Pressure drop value [Pa]'].append(loc_90_H*mass_flow_ISC**2)
    
    # ISC cold leg bends
    data['Pressure drop location'].append('ISC - Cold leg 90 deg bends (per bend)')
    data['Pressure drop value [Pa]'].append(loc_90_C*mass_flow_ISC**2)
    
    # ISC distributed losses hot leg
    data['Pressure drop location'].append('ISC - Distributed hot leg losses')
    data['Pressure drop value [Pa]'].append(dist_H_leg*mass_flow_ISC**2)
    
    # ISC distributed losses cold leg
    data['Pressure drop location'].append('ISC - Distributed cold leg losses')
    data['Pressure drop value [Pa]'].append(dist_C_leg*mass_flow_ISC**2)
    
    # HX1 losses
    data['Pressure drop location'].append('HX1-ISC - Shell losses')
    data['Pressure drop value [Pa]'].append(Shell_HX1_losses*mass_flow_ISC**2)
    
    # HX2 total losses
    data['Pressure drop location'].append('HX2-ISC - Total losses')
    data['Pressure drop value [Pa]'].append(HX2_losses*mass_flow_ISC**2)
    
    # HX1 local losses
    data['Pressure drop location'].append('HX2-ISC - Local losses')
    data['Pressure drop value [Pa]'].append(HX2_loc_losses*mass_flow_ISC**2)
    
    # HX1 distributed losses
    data['Pressure drop location'].append('HX2-ISC - Distributed losses')
    data['Pressure drop value [Pa]'].append(HX2_distrubed_losses*mass_flow_ISC**2)

    
    # ================================
    # PSC ----------------------------
    # ================================    
    # Iterative quantities
    err_out = 1
    toll_out = 1.e-5
    cc_out = 0
    COUNT = 1e4
    
    # External quantities
    h_ISC_HX1 = rr.h_HX1(Re_shell,dh.pitch_HX1,dh.Dout_HX1,Pr_shell,kth_shell)
    Din_HX1 = dh.Dout_HX1 - 2*dh.thick_HX1
    q2_HX1 = dh.Pth/dh.Atransfer_HX1
    
    # Leg fixed parameter 
    _, Din_PSC = pp.get_diameter(pp.get_row(dh.Dout_PSC), col='100')
    
    # Initial guesses
    mass_flow = 100     # kg/s
    T_hot = 160         # °C
    
    while err_out > toll_out and cc_out < COUNT:
        # print(f'{cc_out} iter: mass_rate = {mass_flow:.4f}, relative err: {err_out:.3e}')
        
        # Iterative quantities
        err_in = 1
        toll_in = 1.e-5
        cc_in = 0
        
        # Temperature independent parameters
        Delta_h = dh.Pth/mass_flow
        mass_flow_pipe_HX1 = mass_flow/dh.Ntubes_HX1

        while err_in > toll_in and cc_in < COUNT:
            # print(f'{cc_in} iter: T hot = {T_hot:.4f}, relative err: {err_in:.3e}')
            
            # Fluid states definition
            w_hot = water.with_state(Input.temperature(T_hot), Input.pressure(dh.p_PSC))
            h_hot = w_hot.enthalpy
            h_cold = h_hot - Delta_h
            w_cold = water.with_state(Input.pressure(dh.p_PSC), Input.enthalpy(h_cold))
            T_cold = w_cold.temperature
            T_avg = (T_hot + T_cold)/2
            w_avg = water.with_state(Input.temperature(T_avg), Input.pressure(dh.p_PSC))
            Delta_T = T_hot - T_cold
            
            # Fluid properties in HX1 (T_avg)         
            Nu = rr.Nusselt(w_avg, mass_flow_pipe_HX1, Din_HX1, dh.epsilon_rel_HX1)
            k_w = w_avg.conductivity
            cp_w = w_avg.specific_heat
            
            h_in_HX1 = Nu * k_w / Din_HX1
            
            # Solving HX1
            inverse_Uw_HX1 = dh.Dout_HX1/Din_HX1/h_in_HX1 + dh.Dout_HX1*m.log(dh.Dout_HX1/Din_HX1)/2/dh.k_HX1 + 1/h_ISC_HX1
            Uw_HX1 = pow(inverse_Uw_HX1 ,-1)
            NTUw_HX1 = Uw_HX1*dh.Atransfer_HX1/mass_flow/cp_w
            
            DeltaT_ML_HX1 = q2_HX1 / Uw_HX1 / dh.FT_HX1
            
            # Fining the new T_hot
            T_hot_new = T_hot_ISC + DeltaT_ML_HX1*(dh.FT_HX1*NTUw_HX1*(Delta_T/(T_hot_ISC - T_cold_ISC) -1))/(1-m.exp(-dh.FT_HX1*NTUw_HX1*(Delta_T/(T_hot_ISC - T_cold_ISC) -1)))
            
            # Updating loop parameters
            cc_in += 1
            err_in = abs(T_hot-T_hot_new)/T_hot
            
            T_hot = T_hot_new
        
        # print(f'{cc_in} iter: T hot = {T_hot:.4f}, relative err: {err_in:.3e}')
        
        # Re-definition of fluids
        w_hot = water.with_state(Input.temperature(T_hot), Input.pressure(dh.p_PSC))
        h_hot = w_hot.enthalpy
        h_cold = h_hot - Delta_h
        w_cold = water.with_state(Input.pressure(dh.p_PSC), Input.enthalpy(h_cold))
        T_cold = w_cold.temperature
        T_avg = (T_hot + T_cold)/2
             
        # Properties of hot water
        rho_hot = w_hot.density
        Re_hot = rr.Reynolds_massrate(w_hot,mass_flow,Din_PSC)
        # Properties of cold water
        rho_cold = w_cold.density
        Re_cold = rr.Reynolds_massrate(w_cold,mass_flow,Din_PSC)
        
        # Losses of hot leg PSC
        # Localized 
        loc_90_H = rr.localized_loss_coeff(dh.k_loss_PSC,rho_hot,Din_PSC)
        # Distributed 
        f_hot = rr.friction_rough(Re_hot,dh.epsilon_rel_PSC)
        dist_H_leg = rr.distributed_loss_coeff(dh.L_PSC/2,Din_PSC,f_hot,rho_hot)
        # Total losses inside the hot leg PSC
        Hot_losses = dh.N_bends_PSC * loc_90_H + dist_H_leg
    
        # Losses of cold leg PSC     
        # Localized 
        loc_90_C = rr.localized_loss_coeff(dh.k_loss_PSC,rho_cold,Din_PSC)
        valve_loss = rr.localized_loss_coeff(dh.k_valve_PSC,rho_cold,Din_PSC)
        # Distributed 
        f_cold = rr.friction_rough(Re_cold,dh.epsilon_rel_PSC,)
        dist_C_leg = rr.distributed_loss_coeff(dh.L_PSC/2,Din_PSC,f_cold,rho_cold)
        # Total losses inside the cold leg PSC
        Cold_losses = dh.N_bends_PSC * loc_90_C + dist_C_leg + valve_loss
        
        # Losses inside the HX1 pipes
        # Fluid parameters whitin pipes
        w_ave = water.with_state(Input.temperature(T_avg), Input.pressure(dh.p_PSC))
        rho_ave = w_ave.density
        Re_pipes_HX1 = rr.Reynolds_massrate(w_ave,mass_flow_pipe_HX1,Din_HX1)
        # Localized pressure drop coefficient
        loc_90_HX1 = rr.localized_loss_coeff(dh.k_loss_PSC,rho_ave,Din_HX1)/dh.Ntubes_HX1**2
        exp_pipe_header = rr.exp_loss(cs.pi*Din_PSC**2/4,dh.Aheader_HX1)
        com_header_HX1 = rr.com_loss(dh.Aheader_HX1,cs.pi*Din_HX1**2/4)
        exp_HX1_header = rr.exp_loss(cs.pi*Din_HX1**2/4,dh.Aheader_HX1)
        com_header_pipe = rr.com_loss(dh.Aheader_HX1,cs.pi*Din_PSC**2/4)
        coeff_exp_pipe_header = 0.5*exp_pipe_header/rho_hot/(cs.pi*Din_PSC**2/4)**2
        coeff_com_header_HX1 = 0.5*com_header_HX1/rho_hot/(cs.pi*Din_HX1**2/4*dh.Ntubes_HX1)**2
        coeff_exp_HX1_header = 0.5*exp_HX1_header/rho_cold/(cs.pi*Din_HX1**2/4*dh.Ntubes_HX1)**2
        coeff_com_header_pipe = 0.5*com_header_pipe/rho_cold/(cs.pi*Din_PSC**2/4)**2
        HX1_local_losses = 2*loc_90_HX1 + coeff_exp_pipe_header + coeff_com_header_HX1 + coeff_exp_HX1_header + coeff_com_header_pipe
        # Distributed pressure drop coefficient
        f_HX1 = rr.friction_rough(Re_pipes_HX1,dh.epsilon_rel_HX1)
        HX1_distrubed_losses = rr.distributed_loss_coeff(dh.Ltubes_HX1,Din_HX1,f_HX1,rho_ave)/dh.Ntubes_HX1**2
        
        HX1_losses = HX1_local_losses + HX1_distrubed_losses
        
        # Losses inside the reactor core
        Core_losses = dh.DeltaP_Vessel/dh.mass_rate_reference**2
        
        # Gravitational potential
        Core_potential = - cs.g * (rho_cold - (rho_cold-rho_hot)/2)* dh.H2_PSC
        Potential = (rho_ave *dh.Dout_HX1 + rho_cold *(dh.H1_PSC + dh.H2_PSC - dh.Dout_HX1) - rho_hot * dh.H1_PSC) * cs.g + Core_potential
        
        # New mass rate
        mass_flow_new = pow(Potential/(Hot_losses + Cold_losses + HX1_losses + Core_losses), 0.5)
        
        # Updating loop parameters
        cc_out += 1
        err_out = abs(mass_flow-mass_flow_new)/mass_flow
            
        mass_flow = mass_flow_new
        
    print('---------------------------------------------------------')
    print(f'PSC iteration complete:\nmass flow rate = {mass_flow:.4f} kg/s\nAverage temperature = {T_avg:.4f} °C\nHot channel temperature = {T_hot:.4f} °C')
    print('---------------------------------------------------------')
    
    mass_flow_PSC = mass_flow
    
    # PSC buoyancy
    data['Pressure drop location'].append('PSC - Driving buoyancy')
    data['Pressure drop value [Pa]'].append(Potential)
    
    # PSC total losses
    data['Pressure drop location'].append('PSC - Total pressure losses')
    data['Pressure drop value [Pa]'].append((Hot_losses + Cold_losses + HX1_losses + Core_losses)*mass_flow_PSC**2)
    
    # Core
    data['Pressure drop location'].append('PSC - Pressure vessel losses')
    data['Pressure drop value [Pa]'].append(Core_losses*mass_flow_PSC**2)
    
    # PSC hot leg bends
    data['Pressure drop location'].append('PSC - Hot leg 90 deg bends (per bend)')
    data['Pressure drop value [Pa]'].append(loc_90_H*mass_flow_PSC**2)
    
    # PSC cold leg bends
    data['Pressure drop location'].append('PSC - Cold leg 90 deg bends (per bend)')
    data['Pressure drop value [Pa]'].append(loc_90_C*mass_flow_PSC**2)
    
    # PSC cold leg valve
    data['Pressure drop location'].append('PSC - Cold leg valve')
    data['Pressure drop value [Pa]'].append(valve_loss*mass_flow_PSC**2)
    
    # PSC distributed losses hot leg
    data['Pressure drop location'].append('PSC - Distributed hot leg losses')
    data['Pressure drop value [Pa]'].append(dist_H_leg*mass_flow_PSC**2)
    
    # PSC distributed losses cold leg
    data['Pressure drop location'].append('PSC - Distributed cold leg losses')
    data['Pressure drop value [Pa]'].append(dist_C_leg*mass_flow_PSC**2)
    
    # HX1 total losses
    data['Pressure drop location'].append('HX1-PSC - Total losses')
    data['Pressure drop value [Pa]'].append(HX1_losses*mass_flow_PSC**2)
    
    # HX1 local losses
    data['Pressure drop location'].append('HX1-PSC - Local losses')
    data['Pressure drop value [Pa]'].append(HX1_local_losses*mass_flow_PSC**2)
    
    # HX1 distributed losses
    data['Pressure drop location'].append('HX1-PSC - Distributed losses')
    data['Pressure drop value [Pa]'].append(HX1_distrubed_losses*mass_flow_PSC**2)
    
    # Output
    pd.DataFrame(data).to_csv('pressure_drops.csv',index=False, float_format='%.4f')
    
    
    