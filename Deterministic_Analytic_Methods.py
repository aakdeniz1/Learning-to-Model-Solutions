'''Import modules'''
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import factorial
from myGillespie(SSA) import choose_tau_mu, binom, gillespie
import matplotlib.pyplot as plt

'''First Order Simulation'''

# Reaction Rate
k1 = 3 # 1/s

# Species Initial Concentrations (μM)
A0 = 100
B0 = 0

# Initialize time vector and number of stochastic trajectories
tspan = 2
tvals = np.linspace(0, tspan, 100)
numtraj = 10

# Analytical Solutions
A = A0*np.exp(-k1*tvals)
B = B0 + A0*(1 - np.exp(-k1*tvals))

# Numerical Solutions
def dAdt(t, A):
    return -k1*A

def dBdt(t, B):
    return k1*(A0 + B0 - B)

A_num = solve_ivp(dAdt, (0, tspan), [A0], t_eval = tvals)
B_num = solve_ivp(dBdt, (0, tspan), [B0], t_eval = tvals)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(14,7))
ax.set_title(f'1st-Order Chemical Reaction A --> B with Rate k1 = {k1} Hz')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Concentration (μM)')

ax.plot(tvals, A, label = 'Analytical A', linewidth=2.0)
ax.plot(tvals, B, label = 'Analytical B', linewidth=2.0)

ax.plot(tvals, A_num.y[0], '--', label = "Numerical A", linewidth=2.0)
ax.plot(tvals, B_num.y[0], '--', label = "Numerical B", linewidth=2.0)

half_times = np.zeros(numtraj)
for i in range(numtraj):
    Xinit = np.array([A0, B0])
    rates = np.array([k1])
    stoch_react = np.array([[-1,0]])
    stoch_prod = np.array([[0,1]])
    tspan = 2
    max_iterations = 10000
    
    results = gillespie(Xinit, rates, stoch_react, stoch_prod, tspan, max_iterations)

    ax.plot(results[0], results[1][:,0], '-.')
    ax.plot(results[0], results[1][:,1], '-.')
    
    for j in range(len(results[1])):
        if results[1][j, 0] == A0/2:
            half_times[i] = results[0][j]
    
ax.legend()
plt.show()

tau_theoretical = np.log(2)/k1
tau_actual = sum(half_times)/numtraj

print(f'Half times for {numtraj} trajectories: {half_times}')
print(f'Avg half time: {tau_actual} seconds')
print(f'Theoretical half time: ln(2)/k1 = {tau_theoretical} seconds')
print(f'Percent deviation: (actual-theoretical)/theoretical = {abs(tau_actual-tau_theoretical)/tau_theoretical*100} %')

'''Second Order Simulation'''

# Reaction Rate
k2 = 1 # /μM/s

# Species Initial Concentrations (μM) and Volume (/μM)
A0 = 1
B0 = 2
C0 = 0

V = 100

# Initialize time vector and number of stochastic trajectories
tspan = 4
tvals = np.linspace(0, tspan, 100)
numtraj = 10

# Analytical Solutions
A = (A0 - B0)/(1 - B0/A0*np.exp(-k2*(A0 - B0)*tvals))
B = (B0 - A0)/(1 - A0/B0*np.exp(-k2*(B0 - A0)*tvals))
C = C0 + A0*(1 - np.exp(-k2*(B0 - A0)*tvals))/(1 - A0/B0*np.exp(-k2*(B0 - A0)*tvals))

# Numerical Solutions
def dAdt(t, A):
    return -k2*A*(B0 - A0 + A)

def dBdt(t, B):
    return -k2*B*(A0 - B0 + B)

def dCdt(t, C):
    return k2*(A0 + C0 - C)*(B0 + C0 - C)

A_num = solve_ivp(dAdt, (0, tspan), [A0], t_eval = tvals)
B_num = solve_ivp(dBdt, (0, tspan), [B0], t_eval = tvals)
C_num = solve_ivp(dCdt, (0, tspan), [C0], t_eval = tvals)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(14,7))
ax.set_title(f'2nd-Order Chemical Reaction A + B --> C with Rate k2 = {k2} /μM/s')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Concentration (μM)')

ax.plot(tvals, A, label = 'Analytical A', linewidth=2.0)
ax.plot(tvals, B, label = 'Analytical B', linewidth=2.0)
ax.plot(tvals, C, label = 'Analytical C', linewidth=2.0)

ax.plot(tvals, A_num.y[0], '--', label = "Numerical A", linewidth=2.0)
ax.plot(tvals, B_num.y[0], '--', label = "Numerical B", linewidth=2.0)
ax.plot(tvals, C_num.y[0], '--', label = "Numerical C", linewidth=2.0)

half_times = np.zeros(numtraj)
for i in range(numtraj):
    Xinit = V*np.array([A0, B0, C0])
    rates = np.array([k2/V])
    stoch_react = np.array([[-1,-1,0]])
    stoch_prod = np.array([[0,0,1]])
    tspan = 4
    max_iterations = 10000

    results = gillespie(Xinit, rates, stoch_react, stoch_prod, tspan, max_iterations)

    store_t = results[0]
    store_X = results[1]/V
    store_Rnum = results[2]

    ax.plot(store_t, store_X[:,0], '-.')
    ax.plot(store_t, store_X[:,1], '-.')
    ax.plot(store_t, store_X[:,2], '-.')
    
    for j in range(len(results[1])):
        if results[1][j, 0] == V*A0/2:
            half_times[i] = results[0][j]

ax.legend()
plt.show()

tau_theoretical = np.log(2 - A0/B0)/(B0-A0)/k2
tau_actual = sum(half_times)/numtraj

print(f'Half times for {numtraj} trajectories: {half_times}')
print(f'Avg half time: {tau_actual} seconds')
print(f'Theoretical half time: ln(2-A0/B0)/(B0-A0)/k2 = {tau_theoretical} seconds')
print(f'Percent deviation: (actual-theoretical)/theoretical = {abs(tau_actual-tau_theoretical)/tau_theoretical*100} %')
