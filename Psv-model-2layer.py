import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

def format_to_scientific(value):
    return "{:.5e}".format(value)

def leading_zero(value):
    return '{:02d}'.format(value)

def format_delta(value):
    return "{:.2e}".format(value).replace("+", "").replace("-","")

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
# sl_fact = 1/2
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

start_time = time.time()
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
    ### General update for tauxx
    # tauxx[:,:] = tauxx_n[:,:] + (rhocp2 * dt / dx) * (vx[1:,:] - vx[:-1,:]) + \
    #             (rhocp2m2cs2 * dt / dz) * (vz[:,1:] - vz[:,:-1])
    tauxx[:,:intlz-1] = tauxx_n[:,:intlz-1] + (rhocp2 * dt / dx) * (vx[1:,:intlz-1] - vx[:-1,:intlz-1]) + \
                        (rhocp2m2cs2 * dt / dz) * (vz[:,1:intlz] - vz[:,:intlz-1])
    tauxx[:,intlz-1] = tauxx_n[:,intlz-1] + (rhocp2 * dt / dx) * (vx[1:,intlz-1] - vx[:-1,intlz-1]) + \
                    (rhocp2m2cs2 * dt / dz) * (vz[:,imlz] - vz[:,imlz-1])
    tauxx[:,intlz] = tauxx_n[:,intlz] + (rhocp2 * dt / dx) * (vx[1:,intlz] - vx[:-1,intlz]) + \
                    (rhocp2m2cs2 * dt / dz) * (vz[:,iplz+1] - vz[:,iplz])
    tauxx[:,intlz+1:] = tauxx_n[:,intlz+1:] + (rhocp2 * dt / dx) * (vx[1:,intlz+1:] - vx[:-1,intlz+1:]) + \
                        (rhocp2m2cs2 * dt / dz) * (vz[:,iplz+2:] - vz[:,iplz+1:-1])
                        

    ### General update for tauzz
    # tauzz[:,:] = tauzz_n[:,:] + (rhocp2 * dt / dz) * (vz[:,1:] - vz[:,:-1]) + \
    #             (rhocp2m2cs2 * dt / dx) * (vx[1:,:] - vx[:-1,:])
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
    # tauxz[1:-1,imlz] = tauxz_n[1:-1,imlz] + (rhocs2 * dt / dz) * (vx[1:-1,intlz] - vx[1:-1,intlz-1]) + \
    #                 (rhocs2 * dt / dx) * (vz[1:,imlz] - vz[:-1,imlz])
    tauxz[1:-1,imlz] = tauxz_n[1:-1,imlz] + dt * del1 * (vx[1:-1,intlz] - vx[1:-1,intlz-1])
    ### Z_I^(+) layer
    # tauxz[1:-1,iplz] = tauxz_n[1:-1,iplz] + (rhocs2 * dt / dz) * (vx[1:-1,intlz] - vx[1:-1,intlz-1]) + \
    #                 (rhocs2 * dt / dx) * (vz[1:,iplz] - vz[:-1,iplz])
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


        #Velocity Plot
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # surf = ax.plot_surface(vx_xx, vx_zz, vx, cmap=cm.viridis, rstride=5, cstride=5)
        # cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        # # ax.set_zlim(-6e-10, 6e-10)
        # # surf.set_clim(-6e-10, 6e-10)
        # ax.view_init(35, -65)
        # ax.set_title(rf'$v_x$ at time {format_to_scientific(th[n-1])}s', fontsize=14, color='k')
        # ax.set_xlabel("x [m]", fontsize=14, color='k')
        # ax.set_ylabel("z [m]", fontsize=14, color='k')
        # plt.savefig(f'output/{leading_zero(plot_counter)}vx_3d_{format_to_scientific(th[n-1])}.png', dpi=300, bbox_inches='tight')
        # plt.close()
        #CONTOUR PLOT
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
        # ax.set_zlim(-6e-10, 6e-10) #2e10, 5e10 worked
        # surf.set_clim(-6e-10, 6e-10)
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
        # ax.plot([vz_xx.min(), vz_xx.max()], [z[intlz], z[intlz]], color='black', linestyle='--', linewidth=1)
        # # ax.set_title(rf'$v_z$ at time {format_to_scientific(th[n-1])}s', fontsize=14, color='k')
        # ax.set_xlabel(r"$x \, [m]$", fontsize=14, color='k')
        # ax.set_ylabel(r"$z \, [m]$", fontsize=14, color='k')
        # ax.set_xlim(vz_xx.min(), vz_xx.max())
        # ax.set_ylim(vz_zz.min(), vz_zz.max())
        # plt.savefig(f'output/{leading_zero(plot_counter)}vz_{format_to_scientific(th[n-1])}.png', dpi=300, bbox_inches='tight')
        # plt.close()

        #Stress Plots
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # surf = ax.plot_surface(norm_xx, norm_zz, tauxx, cmap=cm.viridis, rstride=5, cstride=5)
        # ax.set_zlim(-2e-3, 2e-3) #4e-3 - to show less of boundary reflection
        # surf.set_clim(-2e-3, 2e-3)
        # cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        # ax.view_init(35, -65)
        # plt.title(f"tauxx stress at time: {t[n]}s")
        # ax.set_xlabel("x [m]")
        # ax.set_ylabel("z [m]")
        # #vertical line
        # # ax.plot([xh[intlx], xh[intlx]], [norm_zz.min(), norm_zz.max()], color='red', linewidth=1)
        # #horizontal line
        # ax.plot([norm_xx.min(), norm_xx.max()], [zh[intlz], zh[intlz]], color='red', linewidth=1)
        # #staircase lines
        # # ax.plot([xh[intlx], xh[intlx]], [norm_zz.min(), z[intlz]], color='red', linewidth=1)
        # # ax.plot([norm_xx.min(), x[intlx]], [zh[intlz], zh[intlz]], color='red', linewidth=1)
        # plt.savefig(f'output/{leading_zero(plot_counter)}tauxx_{format_to_scientific(t[n])}.png', dpi=300, bbox_inches='tight')
        # plt.close()
        #CONTOUR PLOT
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
        # ax.set_zlim(-4e-3, 4e-3)
        # surf.set_clim(-4e-3, 4e-3)
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
        # ax.set_zlim(-10e-3, 10e-3)
        # surf.set_clim(-10e-3, 10e-3)
        # cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        # ax.view_init(35, -65)
        # plt.title(f"tauxz stress at time: {t[n]}s")
        # ax.set_xlabel("x [m]")
        # ax.set_ylabel("z [m]")
        # plt.savefig(f'output/{leading_zero(plot_counter)}tauxz_{format_to_scientific(t[n])}.png', dpi=300, bbox_inches='tight')
        # plt.close()
        #CONTOUR PLOT
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111)
        # contour = ax.contourf(shear_xx, shear_zz, tauxz, levels=100, vmin=-6e-5, vmax=6e-5, cmap=cm.RdBu)
        # cbar = fig.colorbar(contour, ax=ax, shrink=0.5, aspect=10)
        # # Plot the horizontal fracture line
        # ax.plot([shear_xx.min(), shear_xx.max()], [z[intlz], z[intlz]], color='black', linestyle='--', linewidth=1)
        # # ax.set_title(rf'$\tau_{{xz}}$ at time {format_to_scientific(t[n])}s', fontsize=14, color='k')
        # ax.set_xlabel(r"$x [m]$", fontsize=14, color='k')
        # ax.set_ylabel(r"$z [m]$", fontsize=14, color='k')
        # ax.set_xlim(shear_xx.min(), shear_xx.max())
        # ax.set_ylim(shear_zz.min(), shear_zz.max())
        # plt.savefig(f'output/{leading_zero(plot_counter)}tauxz_{format_to_scientific(t[n])}.png', dpi=300, bbox_inches='tight')
        # plt.close()

        if plot_counter < len(plot_times) - 1:
            plot_counter += 1
        else:
            break
end_time = time.time()
print(f"Elapsed Time = {round((end_time-start_time)/60, 3)}mins")