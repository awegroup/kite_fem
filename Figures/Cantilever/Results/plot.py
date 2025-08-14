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
FEM_timoshenko3 = import_data("FEM_timoshenko3Node.csv")
FEM_timoshenko5 = import_data("FEM_timoshenko5Node.csv")

PSM2D["L_ratio"] 



#L_ratio,Load_param,Force_Angle,Vertical_disp,Horizontal_disp,Beam_angle
L_ratio = np.array(PSM2D["L_ratio"])
Load_param = np.array(PSM2D["Load_param"])
Vertical_disp = np.array(PSM2D["Vertical_disp"])
Horizontal_disp = np.array(PSM2D["Horizontal_disp"])
Beam_angle = np.array(PSM2D["Beam_angle"])

L_ratios = np.unique_values(L_ratio)
fig1, ax1 = plt.subplots(figsize=(4,4))
fig2, ax2 = plt.subplots(figsize=(4,4))
fig3, ax3 = plt.subplots(figsize=(4,4))


for i in range(len(L_ratios)):
    Bool = np.zeros_like(L_ratio,dtype=bool)
    Bool = np.where(L_ratio == L_ratios[i],True,Bool)
    ax1.plot(Vertical_disp[Bool],Load_param[Bool], label = f"PSM, AR = {L_ratios[i]}",color=colors[i],)
    ax2.plot(Horizontal_disp[Bool],Load_param[Bool], label = f"PSM, AR = {L_ratios[i]}",color=colors[i])
    ax3.plot(Beam_angle[Bool],Load_param[Bool], label = f"PSM, AR = {L_ratios[i]}",color=colors[i])

ax1.plot(Literature["Vertical_disp"],Literature["Load_param"], linestyle="-",color="black",label="Mattiasson")
ax2.plot(Literature["Horizontal_disp"],Literature["Load_param"], linestyle="-",color="black",label="Mattiasson")
ax3.plot(Literature["Beam_angle"],Literature["Load_param"], linestyle="-",color="black",label="Mattiasson")



ax1.plot(FEM_timoshenko3["Vertical_disp"],FEM_timoshenko3["Load_param"], linestyle="--",color="gray",label="Timoshenko, N=3")
ax3.plot(FEM_timoshenko3["Beam_angle"],FEM_timoshenko3["Load_param"], linestyle="--",color="gray",label="Timoshenko, N=3")
ax1.plot(FEM_timoshenko5["Vertical_disp"],FEM_timoshenko5["Load_param"], linestyle="--",color="purple",label="Timoshenko, N=5")
ax3.plot(FEM_timoshenko5["Beam_angle"],FEM_timoshenko5["Load_param"], linestyle="--",color="purple",label="Timoshenko, N=5")




ax1.legend(fontsize="small")
ax1.set_xlabel(f"Non-Dimensional Deflection "+r"$w/L$ (-)")
ax1.set_ylabel(f"Load Parameter "+r"$\frac{P L^2}{EI}$ (-)")
ax1.set_xlim(0,1)
ax1.set_ylim(0,10)
ax1.grid()
fig1.tight_layout()
ax2.legend(fontsize="small")
ax2.set_xlabel(f"Non-Dimensional Deflection "+r"$u/L$ (-)")
ax2.set_ylabel(f"Load Parameter "+r"$\frac{P L^2}{EI}$ (-)")
ax2.set_xlim(0,1)
ax2.set_ylim(0,10)
ax2.grid()
fig2.tight_layout()
ax3.legend(fontsize="small")
ax3.set_xlabel(f"Tip Angle "+r"$\theta_0$ (rad)")
ax3.set_ylabel(f"Load Parameter "+r"$\frac{P L^2}{EI}$ (-)")
ax3.set_xlim(0,1.5)
ax3.set_ylim(0,10)
ax3.grid()
fig3.tight_layout()



# 3D Cantilever PSM
# Unpack data
Force_angle = PSM3D["Force_Angle"]
Vertical_disp = PSM3D["Vertical_disp"]
Horizontal_disp = PSM3D["Horizontal_disp"]
Beam_angle = PSM3D["Beam_angle"]
# Plotting 3D PSM Cantilever force angle sweep vs target values
fig4, ax4 = plt.subplots(figsize=(8,4))
ax4.plot(Force_angle, Vertical_disp, color=colors[0],label=r"$w/L$")
ax4.plot([0,90],[0.46326,0.46326], linestyle="--",color=colors[0])
ax4.plot(Force_angle, Horizontal_disp,color=colors[1], label=r"$u/L$")
ax4.plot([0,90],[0.13981,0.13981], linestyle="--",color=colors[1])
ax4.plot(Force_angle, Beam_angle,color=colors[2], label=r"$\theta_0$")
ax4.plot([0,90],[0.72876,0.72876], linestyle="--",color=colors[2])
ax4.grid()
ax4.set_xlabel(r"Force angle $\alpha$ (°)")
ax4.set_ylabel(r"$\mathrm{Non\text{-}dimensional\ deflection}\ w/L,\ u/L\ [-]$" "\n" r"$\mathrm{angle}\ \theta_0\ (rad)$")
ax4.set_xlim(0,90)
ax4.set_ylim(0,1)
ax4.legend(fontsize="small")
fig4.tight_layout()


fig1.savefig(os.path.join(script_dir, '..', 'Figures', 'Cantilever_Vertical_disp.pdf'))
fig2.savefig(os.path.join(script_dir, '..', 'Figures', 'Cantilever_Horizontal_disp.pdf'))
fig3.savefig(os.path.join(script_dir, '..', 'Figures', 'Cantilever_Beam_angle.pdf'))
fig4.savefig(os.path.join(script_dir, '..', 'Figures', 'Cantilever_PSM_Force_angle.pdf'))