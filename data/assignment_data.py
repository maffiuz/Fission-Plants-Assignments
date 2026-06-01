import scipy.constants as cs

# Problem data
# Core parameters
H = 168*cs.inch                 # m, active height
D_fuel_pellet = 0.3225*cs.inch  # m, fuel pellet diameter
D_fuel_rod = 0.374*cs.inch      # m, fuel road outer diameter
t_cladding = 0.0225*cs.inch     # m, cladding thickness
p_coolant = 2250*cs.psi*1e-6    # MPa, coolant pressure


# Zircaloy-4
sigma_yield = 241               # MPa
sigma_ultimate = 413            # MPa

# Fuel characteristics
BU = 60000*1e-3                 # MWd/kg, fuel burnup factor                  
rho_UO2 = 0.955 * 10960         # kg m-3, UO2 pellet density
e_U235 = 0.0445                 # -, UO2 max enrichment
Rr = 0.4                        # -, fuel gas release ratio
c_N2 = 25*1e-6                  # -, N2 impurity mass concentration in fuel
mm_N2 = 28.02                   # g/mol
c_H2O = 75*1e-6                 # -, H2O impurity mass concentration in fuel
mm_H2O = 18.015                 # g/mol
Y = 0.28                        # -, fission yield
Ef = 200*cs.eV*1e6              # J, fission energy release
