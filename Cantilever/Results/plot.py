import os
script_dir = os.path.dirname(os.path.abspath(__file__))


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

colors = ["b","orange","g","r","purple","brown","pink","gray"]

def import_data(file_name):
    """
    Import data from a CSV file.
    """
    path = os.path.join(script_dir, '..', 'Results', file_name)
    data = pd.read_csv(path)
    dictionary = {col: data[col].tolist() for col in data.columns}
    return dictionary
    
PSM2D = import_data("PSM2D.csv")
PSM3D = import_data("PSM3D.csv")
Literature = import_data("Literature.csv")
FEM_timoshenko = import_data("FEM_timoshenko.csv")

PSM2D["L_ratio"] 



#L_ratio,Load_param,Force_Angle,Vertical_disp,Horizontal_disp,Beam_angle
L_ratio = np.array(PSM2D["L_ratio"])
Load_param = np.array(PSM2D["Load_param"])
Vertical_disp = np.array(PSM2D["Vertical_disp"])
Horizontal_disp = np.array(PSM2D["Horizontal_disp"])
Beam_angle = np.array(PSM2D["Beam_angle"])

L_ratios = np.unique_values(L_ratio)
fig1, ax1 = plt.subplots(figsize=(6,6))
fig2, ax2 = plt.subplots(figsize=(6,6))
fig3, ax3 = plt.subplots(figsize=(6,6))


for i in range(len(L_ratios)):
    Bool = np.zeros_like(L_ratio,dtype=bool)
    Bool = np.where(L_ratio == L_ratios[i],True,Bool)
    ax1.scatter(Vertical_disp[Bool],Load_param[Bool], label = f"L_ratio = {L_ratios[i]}",color=colors[i])
    ax2.scatter(Horizontal_disp[Bool],Load_param[Bool], label = f"L_ratio = {L_ratios[i]}",color=colors[i])
    ax3.scatter(Beam_angle[Bool],Load_param[Bool], label = f"L_ratio = {L_ratios[i]}",color=colors[i])

ax1.plot(Literature["Vertical_disp"],Literature["Load_param"], linestyle="--",color=colors[len(L_ratios)],label="Literature")
ax2.plot(Literature["Horizontal_disp"],Literature["Load_param"], linestyle="--",color=colors[len(L_ratios)],label="Literature")
ax3.plot(Literature["Beam_angle"],Literature["Load_param"], linestyle="--",color=colors[len(L_ratios)],label="Literature")

ax1.plot(FEM_timoshenko["Vertical_disp"],FEM_timoshenko["Load_param"], linestyle="--",color=colors[len(L_ratios)+1],label="Timoshenko")
ax3.plot(FEM_timoshenko["Beam_angle"],FEM_timoshenko["Load_param"], linestyle="--",color=colors[len(L_ratios)+1],label="Timoshenko")

ax1.legend()
ax1.set_xlabel(f"Non-Dimensional Deflection "+r"$w/L$ [-]")
ax1.set_ylabel(f"Load Parameter "+r"$\frac{P L^2}{EI}$ [-]")
ax1.set_xlim(0,1)
ax1.set_ylim(0,10)
ax1.grid()
ax2.legend()
ax2.set_xlabel(f"Non-Dimensional Deflection "+r"$u/L$ [-]")
ax2.set_ylabel(f"Load Parameter "+r"$\frac{P L^2}{EI}$ [-]")
ax2.set_xlim(0,1)
ax2.set_ylim(0,10)
ax2.grid()
ax3.legend()
ax3.set_xlabel(f"Tip Angle "+r"$\theta_0$ [rad]")
ax3.set_ylabel(f"Load Parameter "+r"$\frac{P L^2}{EI}$ [-]")
ax3.set_xlim(0,1.5)
ax3.set_ylim(0,10)
ax3.grid()



# 3D Cantilever PSM
# Unpack data
Force_angle = PSM3D["Force_Angle"]
Vertical_disp = PSM3D["Vertical_disp"]
Horizontal_disp = PSM3D["Horizontal_disp"]
Beam_angle = PSM3D["Beam_angle"]
# Plotting 3D PSM Cantilever force angle sweep vs target values
fig4, ax4 = plt.subplots(figsize=(12,6))
ax4.plot(Force_angle, Vertical_disp, color=colors[0],label=r"$w/L$")
ax4.plot([0,90],[0.46326,0.46326], linestyle="--",color=colors[0])
ax4.plot(Force_angle, Horizontal_disp,color=colors[1], label=r"$u/L$")
ax4.plot([0,90],[0.13981,0.13981], linestyle="--",color=colors[1])
ax4.plot(Force_angle, Beam_angle,color=colors[2], label=r"$\theta_0$")
ax4.plot([0,90],[0.72876,0.72876], linestyle="--",color=colors[2])
ax4.grid()
ax4.set_xlabel(r"Force angle $\alpha$ [°]")
ax4.set_ylabel(r"Non-dimensional deflection $w/L$, $u/L$ [-] & angle $\theta_0$ [rad]")
ax4.set_xlim(0,90)
ax4.set_ylim(0,1)
ax4.legend()


fig1.savefig(os.path.join(script_dir, '..', 'Figures', 'Cantilever_Vertical_disp.pdf'))
fig2.savefig(os.path.join(script_dir, '..', 'Figures', 'Cantilever_Horizontal_disp.pdf'))
fig3.savefig(os.path.join(script_dir, '..', 'Figures', 'Cantilever_Beam_angle.pdf'))
fig4.savefig(os.path.join(script_dir, '..', 'Figures', 'Cantilever_PSM_Force_angle.pdf'))