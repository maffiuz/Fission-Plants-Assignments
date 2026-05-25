import scipy.constants as cs

# Problem data
H = 168*cs.inch                 # m, active height
lambda_tr = 0.29e-2             # m, transport length
Dc = lambda_tr/3                # -, diffusion coefficient in the core
Dr = 0.16                       # -, diffusion coefficient in the reflector
Lr = 2.85e-2                    # m, diffusion length in the reflector
delta = Dc/Dr*Lr                # m, reflector savings
heat_in_fuel = 0.974            # -, heat generated in fuel
Fq = 2.6                        # -, heat flux hot channel factor
P = 3400e6                      # W, reactor core heat output
Q2_avg = 199300*cs.Btu/cs.foot**2/3600  # Wm^-2, average heat flux

# Fuel pellets
N_rods = 41448                  # -, number of fuel rods
D_fuel_pellet = 0.3225*cs.inch  # m, fuel pellet diameter
D_fuel_road = 0.374*cs.inch     # m, fuel road outer diameter
t_gap = 0.0065*cs.inch          # m, gap between fuel and cladding
t_cladding = 0.0225*cs.inch     # m, cladding thickness

# Gap conductance
Ross_Stoute_const = 2.54e-5     # m
A_kHe = 0.1763e-2
N_kHe = 0.77163

# Cladding
poisson_zircaloy = 0.43

# Coolant flow
mass_flow_coolant = 106.8*1e6*cs.lb/3600                # kg/s, effective mass flow rate in the core
A_flow_coolant = 41.8*cs.foot**2                        # m^2, effective flow area
p_coolant = 2250*cs.psi                                 # Pa, coolant pressure
T_in_coolant = cs.convert_temperature(535.0, 'F', 'C')  # °C, inlet coolant temperature
pitch = 0.496*cs.inch           # m, core pitch

# Numerics
n_points = 1001
