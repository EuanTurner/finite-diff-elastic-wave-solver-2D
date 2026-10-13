import numpy as np
import matplotlib.pyplot as plt

def format_to_scientific(value):
    return "{:.5e}".format(value)

def initial_displacement(eps, x, xs):
    return 1/(1 + np.exp(-2*(x-xs)/eps))

def initial_velocity(x, xc, eps):
    return 2/eps/(np.exp((x-xc)/eps) + np.exp(-(x-xc)/eps))**2

#initial velocity parameters
xs = 2.5
eps = 0.45

### Simulation Parameters ###
#Domain
Lx = 10 #m

##Material Properties (Granite)##
#Density
rho = 2750 #kg/m^3
#Young's Modulus
E1 = 50 * 10**9
#Wave speed
cs = np.sqrt(E1/rho)
#Shear Modulus
mu = rho * cs**2

##Discretisation##
#Spatial#
nx = 800
x = np.linspace(0, Lx, nx+1)
dx = x[1] - x[0]

#Time#
tend = 0.004 #s
Nt = 30000
t = np.linspace(0, tend, Nt+1)
dt = t[1] - t[0]

#Stability of scheme
stab = dx/cs
if dt > dx/cs:
    print("stability not satisfied")

sig = cs*dt/dx
sig2 = sig * sig
sig3 = sig2 * sig

#Plotting#
max_plots = 40
Nplt = np.ceil(max_plots/tend)
plot_times = [x/Nplt for x in range(1, max_plots + 1)]
plot_counter = 0

##Film coefficient introduction##
interfconds = "i" #p-perfect contact or i-imperfect contact
delta = 2.4e10 #filmcoeff (SH model film coeff scaled by density)
int_fact = 1/2 #position in grid
intlx = int(np.ceil(int_fact*nx))
#interface minus layer
imlx = intlx
#interface plus layer
iplx = intlx+1

#For plotting (including interface double node)
xl1l2 = np.concatenate((x[:intlx+1],x[intlx:]),axis=0)
xl1 = x[:imlx+1]
xl2 = x[iplx-1:]

##Initialise fields##
u = np.zeros(nx+2) #need extra node to account for interface (double node)
u1 = np.zeros_like(u)

#Initial velocity & displacement profiles
u0 = initial_displacement(eps,xl1l2,xs)
g0 = -cs * initial_velocity(xl1l2,xs,eps)

#Boundary condition coefficients
bcs = "SO" #either second or first order SO/FO respectively
#First order Mur
murfO1 = (cs*dt - dx)/(cs*dt + dx)
#Second order Velasco
velsO1 = (6-11*sig+6*sig2-sig3)/6
velsO2 = (6*sig-5*sig2+sig3)/2
velsO3 = (4*sig2-3*sig-sig3)/2
velsO4 = (2*sig-3*sig2+sig3)/6

fig, ax = plt.subplots()

#first time step done separately:
u1[1:imlx] = u0[1:imlx] + 0.5*sig2*(u0[2:imlx+1] - 2*u0[1:imlx] + u0[:imlx-1]) + dt*g0[1:imlx]
u1[iplx+1:-1] = u0[iplx+1:-1] + 0.5*sig2*(u0[iplx+2:] - 2*u0[iplx+1:-1] + u0[iplx:-2]) + dt*g0[iplx+1:-1]
if interfconds=="i":
    u1[imlx] = u0[imlx] + 0.5*(dt**2/dx**2)*(2*delta*dx*(u0[iplx] - u0[imlx]) - 2*cs**2 * (u0[imlx] - u0[imlx-1])) + \
                dt * g0[imlx]
    u1[iplx] = u0[iplx] + 0.5*(dt**2/dx**2)*(2*cs**2*(u0[iplx+1]-u0[iplx]) - 2*delta*dx*(u0[iplx] - u0[imlx])) + \
                dt * g0[iplx]
elif interfconds=='p':
    u1[imlx] = u0[imlx] + 0.5*sig2*(u0[imlx+2]-2*u0[imlx]+u0[imlx-1]) + dt*g0[imlx]
    u1[iplx] = u0[iplx] + 0.5*sig2*(u0[iplx+1]-2*u0[iplx]+u0[iplx-2]) + dt*g0[iplx]
#Boundary conditions computed second
if bcs == "SO":
    u1[0] = velsO1*u0[0] + velsO2*u0[1] + velsO3*u0[2] + velsO4*u0[3]
    u1[-1] = velsO1*u0[-1] + velsO2*u0[-2] + velsO3*u0[-3] + velsO4*u0[-4]
elif bcs == "FO":
    u1[0] = u0[1] + murfO1*(u1[1] - u0[0])
    u1[-1] = u0[-2] + murfO1*(u1[-2] - u0[-1])

unm1 = np.copy(u0)
un = np.copy(u1)

for n in range(2, Nt+1):

    u[1:imlx] = 2*un[1:imlx] + sig2*(un[2:imlx+1] - 2*un[1:imlx] + un[:imlx-1]) - unm1[1:imlx]
    u[iplx+1:-1] = 2*un[iplx+1:-1] + sig2*(un[iplx+2:] - 2*un[iplx+1:-1] + un[iplx:-2]) - unm1[iplx+1:-1]
    if interfconds == "i":
        u[imlx] = 2*un[imlx] + (dt**2/dx**2)*(2*delta*dx*(un[iplx] - un[imlx]) - 2*cs**2 * (un[imlx] - un[imlx-1])) - \
                    unm1[imlx]
        u[iplx] = 2*un[iplx] + (dt**2/dx**2)*(2*cs**2*(un[iplx+1]-un[iplx]) - 2*delta*dx*(un[iplx] - un[imlx])) - \
                    unm1[iplx]
    elif interfconds == "p":
        u[imlx] = 2*un[imlx] + sig2*(un[imlx+2]-2*un[imlx]+un[imlx-1]) - unm1[imlx]
        u[iplx] = 2*un[iplx] + sig2*(un[iplx+1]-2*un[iplx]+un[iplx-2]) - unm1[iplx]
    #Boundary conditions computed second
    if bcs == "SO":
        u[0] = velsO1*un[0] + velsO2*un[1] + velsO3*un[2] + velsO4*un[3]
        u[-1] = velsO1*un[-1] + velsO2*un[-2] + velsO3*un[-3] + velsO4*un[-4]
    elif bcs == "FO":
        u[0] = un[1] + murfO1*(u[1] - un[0])
        u[-1] = un[-2] + murfO1*(u[-2] - un[-1])

    un, unm1 = np.copy(u), np.copy(un)

    if np.abs(t[n] - plot_times[plot_counter])<=10**-7:
        print(f"plot_time = {plot_times[plot_counter]}")

        ax.clear()
        
        ax.plot(xl1,u[:imlx+1],'bo',fillstyle='none',markevery=1,label="AW solution")
        ax.plot(xl2,u[iplx:],'bo',fillstyle='none',markevery=1)
        ax.plot(xl1,initial_displacement(eps,xl1-cs*t[n],xs),'b-',markevery=15,label="D'Alembert solution")
        ax.plot(xl2,initial_displacement(eps,xl2-cs*t[n],xs),'b-',markevery=15)
        ax.axvline(x = xl1l2[imlx], color = 'k', linestyle='dashed', linewidth=0.5)
        ax.axvline(x = xl1l2[iplx], color = 'k', linestyle='dashed', linewidth=0.5)
        ax.set_title(f"time t={(t[n]*1e3):.4f}ms")
        ax.legend()
        plt.pause(0.1)

        if plot_counter < len(plot_times) - 1:
            plot_counter += 1
        else:
            break






