import numpy as np
import matplotlib.pyplot as plt

#Miscellaneous Functions#
def format_to_scientific(value):
    return "{:.5e}".format(value)

def initial_displacement(eps, x, xs):
    return 1/(1 + np.exp(-2*(x-xs)/eps))

def initial_velocity(eps, x, xs):
    return 2/eps/(np.exp((x-xs)/eps) + np.exp(-(x-xs)/eps))**2

def exact_velocity(eps, x, t, xs, cs):
    return -cs/(2*eps*np.cosh((x-xs-cs*t)/eps)**2)

#initial velocity & displacement parameters
xs = 2.5
eps = 0.45

### Simulation Parameters ###
#Domain
Lx, Lz = 10, 10 #m

##Material Properties (Granite)##
#Density
rho = 2750 #kg/m^3
#Young's Modulus
E1 = 50 * 10**9
#Wave speed
cs = np.sqrt(E1/rho)
#Shear Modulus
mu = E1 #rho * cs**2

##Discretisation##
#Spatial#
nx = 800
x = np.linspace(0, Lx, nx+1)
xh = 0.5*(x[1:]+x[:-1]) #half grid points in x
dx = x[1] - x[0]

#Time#
tend = 0.004 #s
Nt = 30000
t = np.linspace(0, tend, Nt+1)
th = 0.5*(t[1:]+t[:-1]) #half time steps
dt = t[1] - t[0]

factor = 0.95/cs/np.sqrt(2)
stab = (0.95 * dx)/(cs * np.sqrt(2))
if dt > stab:
    print("Unstable")

#Film coefficient introduction
interfconds = "i" #p-perfect contact or i-imperfect contact
delta = 8.25e11 
int_fact = 1/2
intlx = int(np.ceil(int_fact*nx))
#interface minus layer
iml = intlx
#interface plus layer
ipl = intlx+1

delform = "{:.2e}".format(delta/rho).replace("+","")

xl1l2 = np.concatenate((x[:intlx+1],x[intlx:]),axis=0)
xl1 = x[:iml+1]
xl2 = x[ipl-1:]

#Plotting#
max_plots = 40
Nplt = np.ceil(max_plots/tend)
plot_times = [x/Nplt for x in range(1, max_plots + 1)]
plot_counter = 0

##Initialise fields##
vy_n = np.zeros(nx+2)
tauxy_n = np.zeros(nx)
uy_n = np.zeros_like(vy_n)

#initial profiles
vy_n = -cs * initial_velocity(eps, xl1l2, xs) / cs
tauxy_n = mu * initial_velocity(eps, xh, xs) / cs
uy_n = initial_displacement(eps,xl1l2,xs) / cs

##Fields used for update##
vy = np.zeros_like(vy_n)
tauxy = np.zeros_like(tauxy_n)
uy = np.zeros_like(uy_n)

B1 = (dx - cs*dt)/(dx + cs*dt)
B2 = (2*dt)/(rho*(dx + cs*dt))

BSO1 = (4*dx - 3*cs*dt)/(4*dx + 3*cs*dt)
BSO2 = (cs*dt)/(4*dx + 3*cs*dt)
BSO3 = (4*dt)/(rho*(4*dx + 3*cs*dt))

#2-norm max error between D'alembert and numerical 
l2err_max = 0

plot_data = {}
l2err_data = {}

plot_data['xl1'] = xl1
plot_data['xl2'] = xl2

fig, ax = plt.subplots()

#Remaining Time steps
for n in range(1, Nt+1):

    # Left of I minus layer
    vy[1:iml] = vy_n[1:iml] + (dt/rho/dx) * (tauxy_n[1:iml] - tauxy_n[:iml-1]) 
    # Right of I plus later
    vy[ipl+1:-1] = vy_n[ipl+1:-1] + (dt/rho/dx) * (tauxy_n[ipl:] - tauxy_n[iml:-1])

    #INTERFACE
    if interfconds == 'p':
        # I minus layer
        vy[iml] = vy_n[iml] + (dt/rho/dx) * (tauxy_n[iml] - tauxy_n[iml-1])
        # I plus layer
        vy[ipl] = vy_n[ipl] + (dt/rho/dx) * (tauxy_n[iml] - tauxy_n[iml-1])
    elif interfconds == "i":
        # I minus layer #Removed dx scale (remember)
        vy[iml] = vy_n[iml] + (2*dt/rho/dx) * (delta*(uy_n[ipl] - uy_n[iml]) - tauxy_n[iml-1])
        # I plus layer
        vy[ipl] = vy_n[ipl] + (2*dt/rho/dx) * (tauxy_n[iml] - delta*(uy_n[ipl] - uy_n[iml]))

    #Update velocities on boundaries next
    vy[0] = BSO1 * vy_n[0] + BSO2 * (vy[1]+vy_n[1]) + BSO3 * tauxy_n[0]
    vy[-1] = BSO1 * vy_n[-1] + BSO2 * (vy[-2]+vy_n[-2]) - BSO3 * tauxy_n[-1]

    #Update displacement based on velocity
    uy[:] = uy_n[:] + dt*vy[:]

    # Left of I minus layer
    tauxy[:iml-1] = tauxy_n[:iml-1] + ((mu * dt) / dx) * (vy[1:iml] - vy[:iml-1])
    # Right of I plus layer
    tauxy[ipl:] = tauxy_n[ipl:] + ((mu * dt) / dx) * (vy[ipl+2:] - vy[ipl+1:-1])
    # I minus layer
    tauxy[iml-1] = tauxy_n[iml-1] + ((mu * dt) / dx) * (vy[iml] - vy[iml-1])
    # I plus layer
    tauxy[iml] = tauxy_n[iml] + ((mu * dt) / dx) * (vy[ipl+1] - vy[ipl])

    vy_n, tauxy_n, uy_n = np.copy(vy), np.copy(tauxy), np.copy(uy)

    if np.abs(t[n] - plot_times[plot_counter])<=10**-7:
        print(f"plot_time = {plot_times[plot_counter]}")

        ax.clear()

        ax.plot(xl1,cs*uy[:iml+1],'bo',fillstyle='none',markevery=1,label="SH solution")
        ax.plot(xl2,cs*uy[ipl:],'bo',fillstyle='none',markevery=1)
        ax.plot(xl1,initial_displacement(eps,xl1-cs*t[n],xs),'b-',markevery=15,label="D'Alembert solution")
        ax.plot(xl2,initial_displacement(eps,xl2-cs*t[n],xs),'b-',markevery=15)
        ax.axvline(x = xl1l2[iml], color = 'k', linestyle='dashed', linewidth=0.5)
        ax.axvline(x = xl1l2[ipl], color = 'k', linestyle='dashed', linewidth=0.5)
        ax.set_title(f"time t={(t[n]*1e3):.4f}ms")
        ax.legend()
        plt.pause(0.1)

        if plot_counter < len(plot_times) - 1:
            plot_counter += 1
        else:
            break
                    




