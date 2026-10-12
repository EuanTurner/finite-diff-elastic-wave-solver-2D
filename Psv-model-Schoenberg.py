import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def format_to_scientific(value):
    return "{:.5e}".format(value)

def leading_zero(value):
    return '{:02d}'.format(value)

plt.rcParams["text.usetex"] = True

#MATERIAL PARAMETERS
cp = 4500 #m/s
cs = 2500 #m/s
rho = 2750 #kg/m^3
lam = rho * (cp ** 2 - 2 * cs ** 2)
mu = rho * cs ** 2

#CONSTANTS
rhocp2 = rho * cp ** 2 
rhocs2 = rho * cs ** 2
rhocp2m2cs2 = rho * (cp ** 2 - 2 * cs ** 2)

#DOMAIN PARAMETERS
Lx = 1000.0 #m
Lz = 1000.0 #m

#SPATIAL DISCRETISATION
nx, nz = 1000, 1000
dz = Lz/nz
dx = Lx/nx

z = np.linspace(0, Lz, nz+1)
zh = 0.5*(z[1:] + z[:-1]) 

x = np.linspace(0, Lx, nx+1)
xh = 0.5*(x[1:] + x[:-1])

#GRIDS FOR PLOTTING
vx_xx, vx_zz = np.meshgrid(x, zh, indexing='ij') #x-component of velocity
vz_xx, vz_zz = np.meshgrid(xh, z, indexing='ij') #z-component of velocity
norm_xx, norm_zz = np.meshgrid(xh, zh, indexing='ij') #normal stress i.e., tauxx & tauzz
shear_xx, shear_zz = np.meshgrid(x, z, indexing='ij') #shear stress i.e., tauxz

#TIME DISCRETISATION
dt = 1.5e-4 #(0.95 * dx) / (cp * np.sqrt(2)) #based on stability criterion #7.5e-05
t0 = 0 #s
tend = 0.4 #s
Nt = int(np.ceil((tend - t0) / dt))
t = np.linspace(t0, tend, Nt+1)
th = 0.5*(t[1:]+t[:-1]) #half time steps for velocity update

#PLOT VARIABLES
max_plots = 20
Nplt = np.ceil(max_plots/tend)
plot_times = [x/Nplt for x in range(1, max_plots + 1)]
plot_counter = 0

#INITIAL CONDITION (SOURCE)
#Gaussian Wave
def gauss_wave(t,f0,ts):
    return -8*f0*(t-ts)*np.exp(-16*f0**2 * (t-ts)**2)
#gauss wave parameters
f0 = 40 #Hz (s^-1)
ts = 1/f0 #s (centre)
#source location (index for array)
sl_fact = 1/2
slx = int(np.ceil(1/2*nx))
slz = int(np.ceil(1/4*nz))

#BCs constants (second order)
#vz nodes on top and bottom of grid
vzTB1 = (4 * dz - 3 * cp * dt) / (4 * dz + 3 * cp * dt)
vzTB2 = (4 * dz * dt) / (rho * dx * (4 * dz + 3 * cp * dt))
vzTB3 = (4 * dt) / (rho * (4 * dz + 3 * cp * dt))
vzTB4 = (cp * dt) / (4 * dz + 3 * cp * dt)
#tauxz on top and bottom of grid 
tauxzTB1 = (4 * rho * cs * dz - 3 * mu * dt) / (4 * rho * cs * dz + 3 * mu * dt)
tauxzTB2 = (mu * dt) / (4 * rho * cs * dz + 3 * mu * dt)
tauxzTB3 = (4 * rho * cs * mu * dt) / (4 * rho * cs * dz + 3 * mu * dt)
tauxzTB4 = (4 * rho * cs * dz * mu * dt) / (dx * (4 * rho * cs * dz + 3 * mu * dt))
#tauxz on left and right of grid
tauxzLR1 = (4 * rho * cs * dx - 3 * mu * dt) / (4 * rho * cs * dx + 3 * mu * dt)
tauxzLR2 = (4 * rho * cs * dx * mu * dt) / (dz * (4 * rho * cs * dx + 3 * mu * dt))
tauxzLR3 = (4 * rho * cs * mu * dt) / (4 * rho * cs * dx + 3 * mu * dt)
tauxzLR4 = (mu * dt) / (4 * rho * cs * dx + 3 * mu * dt)
#vx on left and right of grid
vxLR1 = (4 * dx - 3 * cp * dt) / (4 * dx + 3 * cp * dt)
vxLR2 = (4 * dt) / (rho * (4 * dx + 3 * cp * dt))
vxLR3 = (cp * dt) / (4 * dx + 3 * cp * dt)
vxLR4 = (4 * dx * dt) / (rho * dz * (4 * dx + 3 * cp * dt))
#tauxz on the corners of grid
tauxzC1 = (dx * (4 * rho * cs * dz - 3 * mu * dt) - dz * 3 * mu * dt) / (dx * (4 * rho * cs * dz + 3 * mu * dt) + dz * 3 * mu * dt)
tauxzC2 = (4 * rho * cs * dx * mu * dt) / (dx * (4 * rho * cs * dz + 3 * mu * dt) + dz * 3 * mu * dt)
tauxzC3 = (mu * dt * dx) / (dx * (4 * rho * cs * dz + 3 * mu * dt) + dz * 3 * mu * dt)
tauxzC4 = (4 * rho * cs * dz * mu * dt) / (dx * (4 * rho * cs * dz + 3 * mu * dt) + dz * 3 * mu * dt)
tauxzC5 = (mu * dt * dz) / (dx * (4 * rho * cs * dz + 3 * mu * dt) + dz * 3 * mu * dt)

#Fracture properties
int_fact = 1/2 
intlz = int(np.ceil(int_fact*nz))

Zn = 10 * dz / (lam + 2 * mu)
Zt = 20 * dz / mu

datfile_title = "10Zn_20Zt" #change for data file

del1 = 1/Zt
del2 = 1/Zn
print(f"Scientific: delta1={format_to_scientific(1/Zn)} and delta2={format_to_scientific(1/Zt)}")
r = lam / (lam + 2 * mu)
delT = Zt * mu / (dz + Zt * mu)
delN = Zn * (lam + 2 * mu) / (dz + Zn * (lam + 2 * mu))

#INITIALISE FIELDS
#Velocity
vx_n = np.zeros((nx+1,nz))
vz_n = np.zeros((nx,nz+1))

#Normal stress
tauxx_n = np.zeros((nx,nz))
tauzz_n = np.zeros((nx,nz))

#Shear stress
tauxz_n = np.zeros((nx+1,nz+1))

#FIELDS FOR UPDATE
vx = np.zeros_like(vx_n)
vz = np.zeros_like(vz_n)
tauxx = np.zeros_like(tauxx_n)
tauzz = np.zeros_like(tauzz_n)
tauxz = np.zeros_like(tauxz_n)

l2err_datavz = {}
l2err_datatxz = {}
plot_data = {}
delta_plot_data = {}
delta_plot_data['shear_xx'] = np.copy(shear_xx)
delta_plot_data['shear_zz'] = np.copy(shear_zz)
plot_data['z'] = z[:]
plot_data['zh'] = zh[:]

#TIME UPDATE
for n in range(1, Nt+1):
    #Internal nodes computed first
    #Velocities computed first
    vx[1:-1,:] = vx_n[1:-1,:] + (dt/rho/dx) * (tauxx_n[1:,:] - tauxx_n[:-1,:]) + \
                (dt/rho/dz) * (tauxz_n[1:-1,1:] - tauxz_n[1:-1,:-1])
    
    vz[:,1:-1] = vz_n[:,1:-1] + (dt/rho/dz) * (tauzz_n[:,1:] - tauzz_n[:,:-1]) + \
                (dt/rho/dx) * (tauxz_n[1:,1:-1] - tauxz_n[:-1,1:-1])

    #Velocity edge conditions (second order)
    #Top Edge vz
    vz[:,-1] = vzTB1 * vz_n[:,-1] + vzTB2 * (tauxz_n[1:,-1] - tauxz_n[:-1,-1]) + \
               vzTB4 * (vz[:,-2] + vz_n[:,-2]) - vzTB3 * tauzz_n[:,-1]
    #Bottom Edge vz
    vz[:,0] = vzTB1 * vz_n[:,0] + vzTB2 * (tauxz_n[1:,0] - tauxz_n[:-1,0]) + \
              vzTB3 * tauzz_n[:,0] + vzTB4 * (vz[:,1] + vz_n[:,1])
    #Left Edge vx
    vx[0,:] = vxLR1 * vx_n[0,:] + vxLR2 * tauxx_n[0,:] + vxLR3 * (vx[1,:] + vx_n[1,:]) + \
              vxLR4 * (tauxz_n[0,1:] - tauxz_n[0,:-1])
    #Right Edge vx
    vx[-1,:] = vxLR1 * vx[-1,:] + vxLR3 * (vx[-2,:] + vx_n[-2,:]) - vxLR2 * tauxx_n[-1,:] + \
               vxLR4 * (tauxz_n[-1,1:] - tauxz_n[-1,:-1])
    
    tauxx[:,:intlz] = tauxx_n[:,:intlz] + (rhocp2 * dt / dx) * (vx[1:,:intlz] - vx[:-1,:intlz]) + \
                     (rhocp2m2cs2 * dt / dz) * (vz[:,1:intlz+1] - vz[:,:intlz])
    tauxx[:,intlz] = tauxx_n[:,intlz] + (dt/dx) * (lam + 2 * mu) * (1 - r ** 2 * delN) * (vx[1:,intlz] - vx[:-1,intlz]) + \
                     (dt/dz) * lam * (1 - delN) * (vz[:,intlz+1] - vz[:,intlz])
    tauxx[:,intlz+1:] = tauxx_n[:,intlz+1:] + (rhocp2 * dt / dx) * (vx[1:,intlz+1:] - vx[:-1,intlz+1:]) + \
                     (rhocp2m2cs2 * dt / dz) * (vz[:,intlz+2:] - vz[:,intlz+1:-1])
    
    
    tauzz[:,:intlz] = tauzz_n[:,:intlz] + (rhocp2 * dt / dz) * (vz[:,1:intlz+1] - vz[:,:intlz]) + \
                (rhocp2m2cs2 * dt / dx) * (vx[1:,:intlz] - vx[:-1,:intlz])
    tauzz[:, intlz] = tauzz_n[:, intlz] + (dt/dx) * lam * (1 - delN) * (vx[1:,intlz] - vx[:-1,intlz]) + \
                      (dt/dz) * (lam + 2 * mu) * (1 - delN) * (vz[:,intlz+1] - vz[:,intlz])
    tauzz[:,intlz+1:] = tauzz_n[:,intlz+1:] + (rhocp2 * dt / dz) * (vz[:,intlz+2:] - vz[:,intlz+1:-1]) + \
                (rhocp2m2cs2 * dt / dx) * (vx[1:,intlz+1:] - vx[:-1,intlz+1:])
    
    
    tauxz[1:-1,1:intlz] = tauxz_n[1:-1,1:intlz] + (rhocs2 * dt / dz) * (vx[1:-1,1:intlz] - vx[1:-1,:intlz-1]) + \
                        (rhocs2 * dt / dx) * (vz[1:,1:intlz] - vz[:-1,1:intlz])
    tauxz[1:-1,intlz] = tauxz_n[1:-1,intlz] + (dt/dx) * mu * (1 - delT) * (vz[1:,intlz] - vz[:-1,intlz]) + \
                        (dt/dz) * mu * (1 - delT) * (vx[1:-1,intlz] - vx[1:-1,intlz-1])
    tauxz[1:-1,intlz+1:-1] = tauxz_n[1:-1,intlz+1:-1] + (rhocs2 * dt / dz) * (vx[1:-1,intlz+1:] - vx[1:-1,intlz:-1]) + \
                        (rhocs2 * dt / dx) * (vz[1:,intlz+1:-1] - vz[:-1,intlz+1:-1])
    
    #Stresses edge conditions (second order)
    #Top edge tauxz
    tauxz[1:-1,-1] = tauxzTB1 * tauxz_n[1:-1,-1] + tauxzTB2 * (tauxz[1:-1,-2] + tauxz_n[1:-1,-2]) - \
                     tauxzTB3 * vx[1:-1,-1] + tauxzTB4 * (vz[1:,-1] - vz[:-1,-1])
    
    #Bottom edge tauxz
    tauxz[1:-1,0] = tauxzTB1 * tauxz_n[1:-1,0] + tauxzTB2 * (tauxz[1:-1,1] + tauxz_n[1:-1,1]) + \
                    tauxzTB3 * vx[1:-1,0] + tauxzTB4 * (vz[1:,0] - vz[:-1,0])
    
    #Left edge tauxz
    tauxz[0,1:-1] = tauxzLR1 * tauxz_n[0,1:-1] + tauxzLR2 * (vx[0,1:] - vx[0,:-1]) + \
                    tauxzLR3 * vz[0,1:-1] + tauxzLR4 * (tauxz[1,1:-1] + tauxz_n[1,1:-1])

    #Right edge tauxz
    tauxz[-1,1:-1] = tauxzLR1 * tauxz_n[-1,1:-1] + tauxzLR2 * (vx[-1,1:] - vx[-1,:-1]) + \
                     tauxzLR4 * (tauxz[-2,1:-1] + tauxz_n[-2,1:-1]) - tauxzLR3 * vz[-1,1:-1]

    #Stresses corner conditions (second order)
    # Top left corner
    tauxz[0,-1] = tauxzC1 * tauxz_n[0,-1] + tauxzC3 * (tauxz[0,-2] + tauxz_n[0,-2]) - \
                  tauxzC2 * vx[0,-1] + tauxzC4 * vz[0,-1] + tauxzC5 * (tauxz[1,-1] + tauxz_n[1,-1]) 
    #Top right corner 
    tauxz[-1,-1] = tauxzC1 * tauxz_n[-1,-1] + tauxzC3 * (tauxz[-1,-2] + tauxz_n[-1,-2]) - \
                   tauxzC2 * vx[-1,-1] + tauxzC5 * (tauxz[-2,-1] + tauxz_n[-2,-1]) - tauxzC4 * vz[-1,-1]
    #Bottom right corner 
    tauxz[-1,0] = tauxzC1 * tauxz_n[-1,0] + tauxzC2 * vx[-1,0] + tauxzC3 * (tauxz[-1,1] + tauxz_n[-1,1]) + \
                  tauxzC5 * (tauxz[-2,0] + tauxz_n[-2,0]) - tauxzC4 * vz[-1,0]
    #Bottom left corner
    tauxz[0,0] = tauxzC1 * tauxz_n[0,0] + tauxzC2 * vx[0,0] + tauxzC3 * (tauxz[0,1] + tauxz_n[0,1]) + \
                 tauxzC4 * vz[0,0] + tauxzC5 * (tauxz[1,0] + tauxz_n[1,0])

    #Source term contribution
    tauxx[slx, slz] += (gauss_wave(t[n],f0,ts) - gauss_wave(t[n-1],f0,ts))
    tauzz[slx, slz] += (gauss_wave(t[n],f0,ts) - gauss_wave(t[n-1],f0,ts))

    l2err_datavz[f"{format_to_scientific(th[n-1])}"] = np.copy(vz[slx,:])
    l2err_datatxz[f"{format_to_scientific(t[n])}"] = np.copy(tauxz[slx,:])

    vx_n, vz_n, tauxx_n, tauzz_n, tauxz_n = np.copy(vx), np.copy(vz), np.copy(tauxx), np.copy(tauzz), np.copy(tauxz)

    if np.abs(t[n] - plot_times[plot_counter])<=10**-3:
        print(f"plot_time = {plot_times[plot_counter]}")

        # #Velocity Plot
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # surf = ax.plot_surface(vx_xx, vx_zz, vx, cmap=cm.viridis, rstride=5, cstride=5)
        # cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        # # ax.set_zlim(-5e-9, 2e-9)
        # # surf.set_clim(-5e-9, 2e-9)
        # ax.view_init(35, -65)
        # plt.title(f"vx velocity at time: {th[n-1]}s")
        # ax.set_xlabel("x [m]")
        # ax.set_ylabel("z [m]")
        # plt.savefig(f'output/{leading_zero(plot_counter)}vx_3d_{format_to_scientific(th[n-1])}.png', dpi=300, bbox_inches='tight')
        # plt.close()
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111)
        # contour = ax.contourf(vx_xx, vx_zz, vx, levels=100, vmin=-1.5e-11, vmax=1.5e-11, cmap=cm.RdBu)
        # cbar = fig.colorbar(contour, ax=ax, shrink=0.5, aspect=10)
        # # Plot the horizontal fracture line
        # ax.plot([vx_xx.min(), vx_xx.max()], [zh[intlz], zh[intlz]], color='black', linestyle='--', linewidth=1)
        # ax.set_title(rf'$v_x$ at time {format_to_scientific(th[n-1])}s', fontsize=14, color='k')
        # ax.set_xlabel("x [m]", fontsize=14, color='k')
        # ax.set_ylabel("z [m]", fontsize=14, color='k')
        # ax.set_xlim(vx_xx.min(), vx_xx.max())
        # ax.set_ylim(vx_zz.min(), vx_zz.max())
        # plt.savefig(f'output/{leading_zero(plot_counter)}vx_{format_to_scientific(th[n-1])}.png', dpi=300, bbox_inches='tight')
        # plt.close()

        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # surf = ax.plot_surface(vz_xx, vz_zz, vz, cmap=cm.viridis, rstride=5, cstride=5)
        # cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        # # ax.set_zlim(-5e-9, 2e-9)
        # # surf.set_clim(-5e-9, 2e-9)
        # ax.view_init(35, -65)
        # plt.title(f"vz velocity at time: {th[n-1]}s")
        # ax.set_xlabel("x [m]")
        # ax.set_ylabel("z [m]")
        # plt.savefig(f'output/{leading_zero(plot_counter)}vz_{format_to_scientific(th[n-1])}.png', dpi=300, bbox_inches='tight')
        # plt.close()
        
        #CONTOUR PLOT
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111)
        # contour = ax.contourf(vz_xx, vz_zz, vz, levels=100, vmin=-1.5e-11, vmax=1.5e-11, cmap=cm.RdBu)
        # cbar = fig.colorbar(contour, ax=ax, shrink=0.5, aspect=10)
        # # Plot the horizontal fracture line
        # ax.plot([vz_xx.min(), vz_xx.max()], [zh[intlz], zh[intlz]], color='black', linestyle='--', linewidth=1)
        # # ax.set_title(rf'$v_z$ at time {format_to_scientific(th[n-1])}s', fontsize=14, color='k')
        # ax.set_xlabel(r"$x \, [m]$", fontsize=14, color='k')
        # ax.set_ylabel(r"$z \, [m]$", fontsize=14, color='k')
        # ax.set_xlim(vz_xx.min(), vz_xx.max())
        # ax.set_ylim(vz_zz.min(), vz_zz.max())
        # plt.savefig(f'output/{leading_zero(plot_counter)}vz_{format_to_scientific(th[n-1])}.png', dpi=300, bbox_inches='tight')
        # plt.close()

        # #Stress Plots
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # surf = ax.plot_surface(norm_xx, norm_zz, tauxx, cmap=cm.viridis, rstride=5, cstride=5)
        # ax.set_zlim(-2e-3, 2e-3)
        # surf.set_clim(-2e-3, 2e-3)
        # cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        # ax.view_init(35, -65)
        # plt.title(f"tauxx stress at time: {t[n]}s")
        # ax.set_xlabel("x [m]")
        # ax.set_ylabel("z [m]")
        # ax.plot([norm_xx.min(), norm_xx.max()], [zh[intlz], zh[intlz]], color='red', linewidth=1)
        # plt.savefig(f'output/{leading_zero(plot_counter)}tauxx_{format_to_scientific(t[n])}.png', dpi=300, bbox_inches='tight')
        # plt.close()
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111)
        # contour = ax.contourf(norm_xx, norm_zz, tauxx, levels=100, vmin=-2e-4, vmax=2e-4, cmap=cm.RdBu)
        # cbar = fig.colorbar(contour, ax=ax, shrink=0.5, aspect=10)
        # # Plot the horizontal fracture line
        # ax.plot([norm_xx.min(), norm_xx.max()], [zh[intlz], zh[intlz]], color='black', linestyle='--', linewidth=1)
        # ax.set_title(rf'$\tau_{{xx}}$ at time {format_to_scientific(t[n])}s', fontsize=14, color='k')
        # ax.set_xlabel("x [m]", fontsize=14, color='k')
        # ax.set_ylabel("z [m]", fontsize=14, color='k')
        # ax.set_xlim(norm_xx.min(), norm_xx.max())
        # ax.set_ylim(norm_zz.min(), norm_zz.max())
        # plt.savefig(f'output/{leading_zero(plot_counter)}tauxx_{format_to_scientific(t[n])}.png', dpi=300, bbox_inches='tight')
        # plt.close()

        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # surf = ax.plot_surface(norm_xx, norm_zz, tauzz, cmap=cm.viridis, rstride=5, cstride=5)
        # ax.set_zlim(-5e-3, 5e-3)
        # surf.set_clim(-5e-3, 5e-3)
        # cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        # ax.view_init(35, -65)
        # plt.title(f"tauzz stress at time: {t[n]}s")
        # ax.set_xlabel("x [m]")
        # ax.set_ylabel("z [m]")
        # plt.savefig(f'output/{leading_zero(plot_counter)}tauzz_{format_to_scientific(t[n])}.png', dpi=300, bbox_inches='tight')
        # plt.close()
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111)
        # contour = ax.contourf(norm_xx, norm_zz, tauzz, levels=100, vmin=-2e-4, vmax=2e-4, cmap=cm.RdBu)
        # cbar = fig.colorbar(contour, ax=ax, shrink=0.5, aspect=10)
        # # Plot the horizontal fracture line
        # ax.plot([norm_xx.min(), norm_xx.max()], [zh[intlz], zh[intlz]], color='black', linestyle='--', linewidth=1)
        # ax.set_title(rf'$\tau_{{zz}}$ at time {format_to_scientific(t[n])}s', fontsize=14, color='k')
        # ax.set_xlabel("x [m]", fontsize=14, color='k')
        # ax.set_ylabel("z [m]", fontsize=14, color='k')
        # ax.set_xlim(norm_xx.min(), norm_xx.max())
        # ax.set_ylim(norm_zz.min(), norm_zz.max())
        # plt.savefig(f'output/{leading_zero(plot_counter)}tauzz_{format_to_scientific(t[n])}.png', dpi=300, bbox_inches='tight')
        # plt.close()

        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # surf = ax.plot_surface(shear_xx, shear_zz, tauxz, cmap=cm.viridis, rstride=5, cstride=5)
        # # ax.set_zlim(-2e-2, 2e-2)
        # # surf.set_clim(-2e-2, 2e-2)
        # cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        # ax.view_init(35, -65)
        # plt.title(f"tauxz stress at time: {t[n]}s")
        # ax.set_xlabel("x [m]")
        # ax.set_ylabel("z [m]")
        # plt.savefig(f'output/{leading_zero(plot_counter)}tauxz_{format_to_scientific(t[n])}.png', dpi=300, bbox_inches='tight')
        # plt.close()
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111)
        # contour = ax.contourf(shear_xx, shear_zz, tauxz, levels=100, vmin=-6e-5, vmax=6e-5, cmap=cm.RdBu)
        # cbar = fig.colorbar(contour, ax=ax, shrink=0.5, aspect=10)
        # # Plot the horizontal fracture line
        # ax.plot([shear_xx.min(), shear_xx.max()], [z[intlz], z[intlz]], color='black', linestyle='--', linewidth=1)
        # # ax.set_title(rf'$\tau_{{xz}}$ at time {format_to_scientific(t[n])}s', fontsize=14, color='k')
        # ax.set_xlabel(r"$x \, [m]$", fontsize=14, color='k')
        # ax.set_ylabel(r"$z \, [m]$", fontsize=14, color='k')
        # ax.set_xlim(shear_xx.min(), shear_xx.max())
        # ax.set_ylim(shear_zz.min(), shear_zz.max())
        # plt.savefig(f'output/{leading_zero(plot_counter)}tauxz_{format_to_scientific(t[n])}.png', dpi=300, bbox_inches='tight')
        # plt.close()

        if plot_counter < len(plot_times) - 1:
            plot_counter += 1
        else:
            break