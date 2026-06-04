import matplotlib.pyplot as plt
import data.assignment_data as dh
import numpy as np
import pandas as pd

# Handles the points results (csv)
def split_results(res: str) -> list:
    df = pd.read_csv(res)
    results = [list(df.loc[:][col]) for col in df.keys()]

    return results

# Specific plots for point a
def plots_point_a():
    Pipe_names, D_pipes, computed_h, v_steam = split_results("out/point_a.csv")   
    
    # Look for the first accepteable h (h<L)
    index_h = 0
    for h in computed_h:
        if h < dh.L:
            index_h = computed_h.index(h) - 1
            break
    
    # Slicing the result lists
    Pipe_names = list(Pipe_names)[index_h:]
    D_pipes = D_pipes[index_h:]
    computed_h = computed_h[index_h:]
    
    # h plot
    plt.figure(figsize=(8,5))
    plt.plot(D_pipes, computed_h, "ro--", label="h")
    plt.xticks(D_pipes, Pipe_names)
    plt.axhline(y=dh.L, color='k', linestyle='--', linewidth=2, label="Max pipe lenght")
    plt.xlabel("D [in]")
    plt.ylabel("h [m]")
    plt.legend()
    plt.grid()
    
    # Getting fastest speed possible
    print(f"{Pipe_names[1]}: steam vel = {v_steam[index_h+1]:.3f} m/s")
    
    
# Specific plots for point b
def plots_point_b():
    Pipe_names, D_pipes, computed_h_a, _ = split_results("out/point_a.csv")   
    _, _, computed_h_b, v_steam_b = split_results("out/point_b.csv")
    
    # Look for the first accepteable h (finite)
    index_h = 0
    for h in computed_h_b:
        if np.isfinite(h):
            index_h = computed_h_b.index(h)
            break
    
    # Slicing the result lists
    Pipe_names = list(Pipe_names)[index_h:]
    D_pipes = D_pipes[index_h:]
    computed_h_b = computed_h_b[index_h:]
    computed_h_a = computed_h_a[index_h:]
    v_steam_b = v_steam_b[index_h:]
    
    # h and steam v plot
    plt.figure(figsize=(8,5))
    plt.plot(D_pipes, computed_h_b, "ro-", label="Minimum h")
    plt.xticks(D_pipes, Pipe_names)
    plt.yticks(color='r')
    plt.xlabel("D [in]")
    plt.ylabel("h [m]")
    plt.grid()
    
    plt.twinx()
    plt.plot(D_pipes, v_steam_b, 'g^-', label='Steam v')
    plt.yticks(color='g')
    plt.ylabel("Steam v [m/s]")   

    # h diff plot
    # Plotting the relative difference from the results computed in point a
    # To show that removing the horizontal isn't always beneficial
    Delta_h_rel = np.array(computed_h_b) / np.array(computed_h_a) - 1
    plt.figure(figsize=(8,5))
    plt.plot(D_pipes, Delta_h_rel, "bo--", label="Minimum h")
    plt.xticks(D_pipes, Pipe_names)
    plt.xlabel("D [in]")
    plt.ylabel("$\Delta_h / h_a$ [-]")
    plt.grid('minor')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=1)
    plt.yticks([-1e-2, -1e-1, 0, 1e-2, 1e-1, 1, 10, 100],['$-10^{-1}$', '$-10^{-2}$', '0', '$10^{-2}$', '$10^{-1}$', '1', '$10$', '$100$'])    
    plt.yscale('symlog', linthresh=1e-2)
    
    
# Specific plots for point c
def plots_point_c():
    Pipe_names, D_pipes, computed_h_b, v_steam_b = split_results("out/point_b.csv")   
    _, _, computed_h_c, v_steam_c, Re = split_results("out/point_c.csv")
    
    # Look for the first accepteable h (finite)
    index_h = 0
    for h in computed_h_c:
        if np.isfinite(h):
            index_h = computed_h_c.index(h)
            break
    
    # Slicing the result lists
    Pipe_names = list(Pipe_names)[index_h:]
    D_pipes = D_pipes[index_h:]
    computed_h_b = computed_h_b[index_h:]
    computed_h_c = computed_h_c[index_h:]
    v_steam_b = v_steam_b[index_h:]
    v_steam_c = v_steam_c[index_h:]
    
    # h diff plot
    # Plotting the relative difference from the results computed in point b
    # To show the difference of rough pipes
    Delta_h_rel = np.array(computed_h_c) / np.array(computed_h_b) - 1
    plt.figure(figsize=(8,5))
    plt.plot(D_pipes, Delta_h_rel, "bo--", label="Minimum h")
    plt.xticks(D_pipes, Pipe_names)
    plt.xlabel("D [in]")
    plt.ylabel("$\Delta_h / h_b$ [-]")
    plt.grid('minor')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=1)
    plt.yticks([-1e-3, 0, 1e-3, 1e-2, 1e-1, 1],['$-10^{-3}$', '0', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '1'], color='b')  
    plt.ylim([-1e-3,1])  
    plt.yscale('symlog', linthresh=1e-3)


def plotter():
    plots_point_a()
    plots_point_b() 
    plots_point_c()
    
    plt.show()
    