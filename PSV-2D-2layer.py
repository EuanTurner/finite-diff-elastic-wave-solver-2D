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

#Fracture properties
int_fact = 1/2 
intlz = int(np.ceil(int_fact*nz))

imlz = intlz
iplz = intlz+1

Zn = 100 * dz / (lam + 2 * mu)
Zt = 200 * dz / mu

del1 = 1/Zt
del2 = 1/Zn

print(f"del1={format_to_scientific(del1)}, del2={format_to_scientific(del2)}")

datfile_title = "10Zn_20Zt" #change for data file

#GRIDS FOR PLOTTING
vx_xx, vx_zz = np.meshgrid(x, zh, indexing='ij') #x-component of velocity

vz_xx, vz_zz = np.meshgrid(xh, z, indexing='ij') #z-component of velocity
vz_xx = np.insert(vz_xx, intlz+1, vz_xx[:,intlz], axis=1)
vz_zz = np.insert(vz_zz, intlz+1, vz_zz[:,intlz], axis=1)

norm_xx, norm_zz = np.meshgrid(xh, zh, indexing='ij') #normal stress i.e., tauxx & tauzz

shear_xx, shear_zz = np.meshgrid(x, z, indexing='ij') #shear stress i.e., tauxz
shear_xx = np.insert(shear_xx, intlz+1, shear_xx[:,intlz], axis=1)
shear_zz = np.insert(shear_zz, intlz+1, shear_zz[:,intlz], axis=1)

#TIME DISCRETISATION
dt = 1.5e-4 #(0.95 * dx) / (cp * np.sqrt(2)) #7.5e-05 #based on stability criterion
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

#INITIALISE FIELDS
#Velocity
vx_n = np.zeros((nx+1,nz))
#nz+2 to account for addition of another row of nodes for vz#
vz_n = np.zeros((nx,nz+2))

#Normal stress
tauxx_n = np.zeros((nx,nz))
tauzz_n = np.zeros((nx,nz))

#Shear stress
tauxz_n = np.zeros((nx+1,nz+2))

#displacements 
ux_n = np.zeros_like(vx_n)
uz_n = np.zeros_like(vz_n)

#FIELDS FOR UPDATE
vx = np.zeros_like(vx_n)
ux = np.zeros_like(ux_n)
vz = np.zeros_like(vz_n)
uz = np.zeros_like(uz_n)
tauxx = np.zeros_like(tauxx_n)
tauzz = np.zeros_like(tauzz_n)
tauxz = np.zeros_like(tauxz_n)

fig, ax = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
cbars = [None, None, None, None, None]   # one per used subplot (ax1-ax5)

#TIME UPDATE
for n in range(1, Nt+1):
    #Internal nodes computed first
    #Velocities computed first
    vx[1:-1,:intlz-1] = vx_n[1:-1,:intlz-1] + (dt/rho/dx) * (tauxx_n[1:,:intlz-1] - tauxx_n[:-1,:intlz-1]) + \
                        (dt/rho/dz) * (tauxz_n[1:-1,1:intlz] - tauxz_n[1:-1,:intlz-1])
    
    ### Around Z_I^(-) layer
    vx[1:-1,intlz-1] = vx_n[1:-1,intlz-1] + (dt/rho/dx) * (tauxx_n[1:,intlz-1] - tauxx_n[:-1,intlz-1]) + \
                        (dt/rho/dz) * (tauxz_n[1:-1,imlz] - tauxz_n[1:-1,imlz-1])
    ### Around Z_I^(+) layer
    vx[1:-1,intlz] = vx_n[1:-1,intlz] + (dt/rho/dx) * (tauxx_n[1:,intlz] - tauxx_n[:-1,intlz]) + \
                        (dt/rho/dz) * (tauxz_n[1:-1,iplz+1] - tauxz_n[1:-1,iplz])
    
    vx[1:-1,intlz+1:] = vx_n[1:-1,intlz+1:] + (dt/rho/dx) * (tauxx_n[1:,intlz+1:] - tauxx_n[:-1,intlz+1:]) + \
                        (dt/rho/dz) * (tauxz_n[1:-1,iplz+2:] - tauxz_n[1:-1,iplz+1:-1])


    ### Before Z_I^(-) layer
    vz[:,1:imlz] = vz_n[:,1:imlz] + (dt/rho/dz) * (tauzz_n[:,1:intlz] - tauzz_n[:,:intlz-1]) + \
                    (dt/rho/dx) * (tauxz_n[1:,1:imlz] - tauxz_n[:-1,1:imlz])
    ### Z_I^(-) layer
    vz[:,imlz] = vz_n[:,imlz] + (2 * dt / (rho * dz)) * (del2 * (uz_n[:,iplz] - uz_n[:,imlz]) - tauzz_n[:,intlz-1]) + \
                    (dt/rho/dx) * (tauxz_n[1:,imlz] - tauxz_n[:-1,imlz])
    ### Z_I^(+) layer
    vz[:,iplz] = vz_n[:,iplz] + (2 * dt / (rho * dz)) * (tauzz_n[:,intlz] - del2 * (uz_n[:,iplz] - uz_n[:,imlz])) + \
                    (dt/rho/dx) * (tauxz_n[1:,iplz] - tauxz_n[:-1,iplz])
    ### After Z_I^(+) layer
    vz[:,iplz+1:-1] = vz_n[:,iplz+1:-1] + (dt/rho/dz) * (tauzz_n[:,intlz+1:] - tauzz_n[:,intlz:-1]) + \
                    (dt/rho/dx) * (tauxz_n[1:,iplz+1:-1] - tauxz_n[:-1,iplz+1:-1])

    #Velocity edge conditions (second order)
    #Top Edge vz
    vz[:,-1] = vzTB1 * vz_n[:,-1] + vzTB2 * (tauxz_n[1:,-1] - tauxz_n[:-1,-1]) + \
               vzTB4 * (vz[:,-2] + vz_n[:,-2]) - vzTB3 * tauzz_n[:,-1]
    #Bottom Edge vz
    vz[:,0] = vzTB1 * vz_n[:,0] + vzTB2 * (tauxz_n[1:,0] - tauxz_n[:-1,0]) + \
              vzTB3 * tauzz_n[:,0] + vzTB4 * (vz[:,1] + vz_n[:,1])
    #Left Edge vx (incorporating interface)
    vx[0,:intlz-1] = vxLR1 * vx_n[0,:intlz-1] + vxLR2 * tauxx_n[0,:intlz-1] + vxLR3 * (vx[1,:intlz-1] + vx_n[1,:intlz-1]) + \
                     vxLR4 * (tauxz_n[0,1:imlz] - tauxz_n[0,:imlz-1])
    vx[0,intlz+1:] = vxLR1 * vx_n[0,intlz+1:] + vxLR2 * tauxx_n[0,intlz+1:] + vxLR3 * (vx[1,intlz+1:] + vx_n[1,intlz+1:]) + \
                     vxLR4 * (tauxz_n[0,iplz+2:] - tauxz_n[0,iplz+1:-1])
    vx[0,intlz-1] = vxLR1 * vx_n[0,intlz-1] + vxLR2 * tauxx_n[0,intlz-1] + vxLR3 * (vx[1,intlz-1] + vx_n[1,intlz-1]) + \
                  vxLR4 * (tauxz_n[0,imlz] - tauxz_n[0,imlz-1])
    vx[0,intlz] = vxLR1 * vx_n[0,intlz] + vxLR2 * tauxx_n[0,intlz] + vxLR3 * (vx[1,intlz] + vx_n[1,intlz]) + \
                  vxLR4 * (tauxz_n[0,iplz+1] - tauxz_n[0,iplz])
    #Right Edge vx (incorporating interface)
    vx[-1,:intlz-1] = vxLR1 * vx[-1,:intlz-1] + vxLR3 * (vx[-2,:intlz-1] + vx_n[-2,:intlz-1]) - vxLR2 * tauxx_n[-1,:intlz-1] + \
                      vxLR4 * (tauxz_n[-1,1:imlz] - tauxz_n[-1,:imlz-1])
    vx[-1,intlz+1:] = vxLR1 * vx[-1,intlz+1:] + vxLR3 * (vx[-2,intlz+1:] + vx_n[-2,intlz+1:]) - vxLR2 * tauxx_n[-1,intlz+1:] + \
                      vxLR4 * (tauxz_n[-1,iplz+2:] - tauxz_n[-1,iplz+1:-1])
    vx[-1,intlz-1] = vxLR1 * vx[-1,intlz-1] + vxLR3 * (vx[-2,intlz-1] + vx_n[-2,intlz-1]) - vxLR2 * tauxx_n[-1,intlz-1] + \
                      vxLR4 * (tauxz_n[-1,imlz] - tauxz_n[-1,imlz-1])
    vx[-1,intlz] = vxLR1 * vx[-1,intlz] + vxLR3 * (vx[-2,intlz] + vx_n[-2,intlz]) - vxLR2 * tauxx_n[-1,intlz] + \
                     vxLR4 * (tauxz_n[-1,iplz+1] - tauxz_n[-1,iplz])
    
    #Update displacement based on velocity
    ux[:,:] = ux_n[:,:] + dt*vx[:,:]
    uz[:,:] = uz_n[:,:] + dt*vz[:,:]

    #Stresses computed second
    tauxx[:,:intlz-1] = tauxx_n[:,:intlz-1] + (rhocp2 * dt / dx) * (vx[1:,:intlz-1] - vx[:-1,:intlz-1]) + \
                        (rhocp2m2cs2 * dt / dz) * (vz[:,1:intlz] - vz[:,:intlz-1])
    tauxx[:,intlz-1] = tauxx_n[:,intlz-1] + (rhocp2 * dt / dx) * (vx[1:,intlz-1] - vx[:-1,intlz-1]) + \
                    (rhocp2m2cs2 * dt / dz) * (vz[:,imlz] - vz[:,imlz-1])
    tauxx[:,intlz] = tauxx_n[:,intlz] + (rhocp2 * dt / dx) * (vx[1:,intlz] - vx[:-1,intlz]) + \
                    (rhocp2m2cs2 * dt / dz) * (vz[:,iplz+1] - vz[:,iplz])
    tauxx[:,intlz+1:] = tauxx_n[:,intlz+1:] + (rhocp2 * dt / dx) * (vx[1:,intlz+1:] - vx[:-1,intlz+1:]) + \
                        (rhocp2m2cs2 * dt / dz) * (vz[:,iplz+2:] - vz[:,iplz+1:-1])


    tauzz[:,:intlz-1] = tauzz_n[:,:intlz-1] + (rhocp2 * dt / dz) * (vz[:,1:intlz] - vz[:,:intlz-1]) + \
            (rhocp2m2cs2 * dt / dx) * (vx[1:,:intlz-1] - vx[:-1,:intlz-1])
    tauzz[:,intlz-1] = tauzz_n[:,intlz-1] + (rhocp2 * dt / dz) * (vz[:,imlz] - vz[:,imlz-1]) + \
        (rhocp2m2cs2 * dt / dx) * (vx[1:,intlz-1] - vx[:-1,intlz-1])
    tauzz[:,intlz] = tauzz_n[:,intlz] + (rhocp2 * dt / dz) * (vz[:,iplz+1] - vz[:,iplz]) + \
        (rhocp2m2cs2 * dt / dx) * (vx[1:,intlz] - vx[:-1,intlz])
    tauzz[:,intlz+1:] = tauzz_n[:,intlz+1:] + (rhocp2 * dt / dz) * (vz[:,iplz+2:] - vz[:,iplz+1:-1]) + \
            (rhocp2m2cs2 * dt / dx) * (vx[1:,intlz+1:] - vx[:-1,intlz+1:])


    ### Before Z_I^(-) layer
    tauxz[1:-1,1:imlz] = tauxz_n[1:-1,1:imlz] + (rhocs2 * dt / dz) * (vx[1:-1,1:intlz] - vx[1:-1,:intlz-1]) + \
                        (rhocs2 * dt / dx) * (vz[1:,1:imlz] - vz[:-1,1:imlz])
    ### Z_I^(-) layer
    tauxz[1:-1,imlz] = tauxz_n[1:-1,imlz] + dt * del1 * (vx[1:-1,intlz] - vx[1:-1,intlz-1])
    ### Z_I^(+) layer
    tauxz[1:-1,iplz] = tauxz_n[1:-1,iplz] + dt * del1 * (vx[1:-1,intlz] - vx[1:-1,intlz-1])
    ### After Z_I^(+) layer
    tauxz[1:-1,iplz+1:-1] = tauxz_n[1:-1,iplz+1:-1] + (rhocs2 * dt / dz) * (vx[1:-1,intlz+1:] - vx[1:-1,intlz:-1]) + \
                    (rhocs2 * dt / dx) * (vz[1:,iplz+1:-1] - vz[:-1,iplz+1:-1])
    
    #Stresses edge conditions (second order)
    #Top edge tauxz
    tauxz[1:-1,-1] = tauxzTB1 * tauxz_n[1:-1,-1] + tauxzTB2 * (tauxz[1:-1,-2] + tauxz_n[1:-1,-2]) - \
                     tauxzTB3 * vx[1:-1,-1] + tauxzTB4 * (vz[1:,-1] - vz[:-1,-1])
    #Bottom edge tauxz
    tauxz[1:-1,0] = tauxzTB1 * tauxz_n[1:-1,0] + tauxzTB2 * (tauxz[1:-1,1] + tauxz_n[1:-1,1]) + \
                    tauxzTB3 * vx[1:-1,0] + tauxzTB4 * (vz[1:,0] - vz[:-1,0])
    #Left edge tauxz (incorporating interface)
    tauxz[0,1:imlz] = tauxzLR1 * tauxz_n[0,1:imlz] + tauxzLR2 * (vx[0,1:intlz] - vx[0,:intlz-1]) + \
                      tauxzLR3 * vz[0,1:imlz] + tauxzLR4 * (tauxz[1,1:imlz] + tauxz_n[1,1:imlz])
    tauxz[0,iplz+1:-1] = tauxzLR1 * tauxz_n[0,iplz+1:-1] + tauxzLR2 * (vx[0,intlz+1:] - vx[0,intlz:-1]) + \
                         tauxzLR3 * vz[0,iplz+1:-1] + tauxzLR4 * (tauxz[1,iplz+1:-1] + tauxz_n[1,iplz+1:-1])
    tauxz[0,imlz] = tauxzLR1 * tauxz_n[0,imlz] + tauxzLR2 * (vx[0,intlz] - vx[0,intlz-1]) + \
                    tauxzLR3 * vz[0,imlz] + tauxzLR4 * (tauxz[1,imlz] + tauxz_n[1,imlz])
    tauxz[0,iplz] = tauxzLR1 * tauxz_n[0,iplz] + tauxzLR2 * (vx[0,intlz] - vx[0,intlz-1]) + \
                    tauxzLR3 * vz[0,iplz] + tauxzLR4 * (tauxz[1,iplz] + tauxz_n[1,iplz])
    #right edge tauxz (incorporating interface)
    tauxz[-1,1:imlz] = tauxzLR1 * tauxz_n[-1,1:imlz] + tauxzLR2 * (vx[-1,1:intlz] - vx[-1,:intlz-1]) + \
                       tauxzLR4 * (tauxz[-2,1:imlz] + tauxz_n[-2,1:imlz]) - tauxzLR3 * vz[-1,1:imlz]
    tauxz[-1,iplz+1:-1] = tauxzLR1 * tauxz_n[-1,iplz+1:-1] + tauxzLR2 * (vx[-1,intlz+1:] - vx[-1,intlz:-1]) + \
                          tauxzLR4 * (tauxz[-2,iplz+1:-1] + tauxz_n[-2,iplz+1:-1]) - tauxzLR3 * vz[-1,iplz+1:-1]
    tauxz[-1,imlz] = tauxzLR1 * tauxz_n[-1,imlz] + tauxzLR2 * (vx[-1,intlz] - vx[-1,intlz-1]) + \
                     tauxzLR4 * (tauxz[-2,imlz] + tauxz_n[-2,imlz]) - tauxzLR3 * vz[-1,imlz]
    tauxz[-1,iplz] = tauxzLR1 * tauxz_n[-1,iplz] + tauxzLR2 * (vx[-1,intlz] - vx[-1,intlz-1]) + \
                     tauxzLR4 * (tauxz[-2,iplz] + tauxz_n[-2,iplz]) - tauxzLR3 * vz[-1,iplz]

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

    vx_n, ux_n, vz_n, uz_n, tauxx_n, tauzz_n, tauxz_n = np.copy(vx), np.copy(ux), np.copy(vz), np.copy(uz), np.copy(tauxx), np.copy(tauzz), np.copy(tauxz)

    if np.abs(t[n] - plot_times[plot_counter])<=10**-3:
        print(f"plot_time = {plot_times[plot_counter]}")

        ax1, ax2, ax3, ax4, ax5, ax6 = ax.flat
        # Remove old colorbars
        for cb in cbars:
            if cb is not None:
                cb.remove()
    
        ax1.clear();ax2.clear();ax3.clear()
        ax4.clear();ax5.clear();ax6.clear()

        # --- Velocity vx ---
        cf1 = ax1.contourf(vx_xx,vx_zz,vx,levels=100,cmap=cm.RdBu)
        ax1.plot([vx_xx.min(), vx_xx.max()], [zh[intlz], zh[intlz]], color='black', linestyle='--', linewidth=1)
        ax1.set_title(f"vx velocity at t={th[n-1]:.6f}s", fontsize=10)
        ax1.set_xlabel(rf"$x$", fontsize=10)
        ax1.set_ylabel(rf"$z$", fontsize=10)
        cbars[0] = fig.colorbar(cf1, ax=ax1)

        # --- Velocity vz ---
        cf2 = ax2.contourf(vz_xx,vz_zz,vz,levels=100,cmap=cm.RdBu)
        ax2.plot([vz_xx.min(), vz_xx.max()], [z[intlz], z[intlz]], color='black', linestyle='--', linewidth=1)
        ax2.set_title(f"vz velocity at t={th[n-1]:.6f}s", fontsize=10)
        ax2.set_xlabel(rf"$x$", fontsize=10)
        ax2.set_ylabel(rf"$z$", fontsize=10)
        cbars[1] = fig.colorbar(cf2, ax=ax2)

        # --- Stress tauxx ---
        cf3 = ax3.contourf(norm_xx,norm_zz,tauxx,levels=100,cmap=cm.RdBu)
        ax3.plot([norm_xx.min(), norm_xx.max()], [zh[intlz], zh[intlz]], color='black', linestyle='--', linewidth=1)
        ax3.set_title(f"tauxx stress at t={t[n]:.6f}s", fontsize=10)
        ax3.set_xlabel(rf"$x$", fontsize=10)
        ax3.set_ylabel(rf"$z$", fontsize=10)
        cbars[2] = fig.colorbar(cf3, ax=ax3)

        # --- Stress tauzz ---
        cf4 = ax4.contourf(norm_xx,norm_zz,tauzz,levels=100,cmap=cm.RdBu)
        ax4.plot([norm_xx.min(), norm_xx.max()], [zh[intlz], zh[intlz]], color='black', linestyle='--', linewidth=1)
        ax4.set_title(f"tauzz stress at t={t[n]:.6f}s", fontsize=10)
        ax4.set_xlabel(rf"$x$", fontsize=10)
        ax4.set_ylabel(rf"$z$", fontsize=10)
        cbars[3] = fig.colorbar(cf4, ax=ax4)

        # --- Stress tauxz ---
        cf5 = ax5.contourf(shear_xx,shear_zz,tauxz,levels=100,cmap=cm.RdBu)
        ax5.plot([shear_xx.min(), shear_xx.max()], [z[intlz], z[intlz]], color='black', linestyle='--', linewidth=1)
        ax5.set_title(f"tauxz stress at t={t[n]:.6f}s", fontsize=10)
        ax5.set_xlabel(rf"$x$", fontsize=10)
        ax5.set_ylabel(rf"$z$", fontsize=10)
        cbars[4] = fig.colorbar(cf5, ax=ax5)

        plt.pause(0.01)

        if plot_counter < len(plot_times) - 1:
            plot_counter += 1
        else:
            break