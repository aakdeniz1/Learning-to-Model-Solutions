'''Import modules'''
import math
import numpy as np
import random
import matplotlib.pyplot as plt

'''Sampling Random Numbers'''

# Pt a - c: Sample Nc = 100 and Nc = 100000 random numbers from uniform distribution (URN). Compare histograms
Nc1 = 100
Nc2 = 100000
xmin = 0
xmax = 1

sample1 = np.random.uniform(xmin, xmax, Nc1)
sample2 = np.random.uniform(xmin, xmax, Nc2)

numbins = 50
binsize = (xmax - xmin)/numbins
print(binsize)

counts1, bins1 = np.histogram(sample1, numbins)
counts2, bins2 = np.histogram(sample2, numbins)

plt.hist(sample1, edgecolor = "black", bins = bins1)
plt.title(f'URN Nc = {Nc1}')
plt.show()
plt.hist(sample2, edgecolor = "black", bins = bins2)
plt.title(f'URN Nc = {Nc2}')
plt.show()

# Parts d-e: Sample Nc = 100000 random numbers from normal distribution. Compare histograms

sample3 = np.random.normal(0, 3, Nc1)
sample4 = np.random.normal(0, 3, Nc2)

counts3, bins3 = np.histogram(sample3, numbins)
counts4, bins4 = np.histogram(sample4, numbins)

plt.hist(sample3, edgecolor = "black", bins = bins3)
plt.title(f'Normal Nc = {Nc1}')
plt.show()
plt.hist(sample4, edgecolor = "black", bins = bins4)
plt.title(f'Normal Nc = {Nc2}')
plt.show()


'''1D Hopping Process'''

# 1D Hop function
def Hop_1D(x0, k, dt, Nsteps):
    pleft = k*dt
    pright = k*dt
    pstay = 1 - (pleft + pright)
    
    step_options = [-1, 0, 1]
    weights = [pleft, pstay, pright] 
    
    hop = np.zeros(Nsteps + 1)
    hop[0] = x0  
    for i in range(Nsteps):
        hop[i+1] = hop[i] + np.random.choice(step_options, p = weights)
        
    return hop

# Run Simulation
Nsteps = 100
Nsims = 1000

x_half = np.zeros(Nsims)
x_end = np.zeros(Nsims)

for i in range(Nsims):
    traj = Hop_1D(0, 1, 0.1, Nsteps)
    x_half[i] = traj[round(Nsteps/2) + 1]
    x_end[i] = traj[Nsteps]

x_half_mean = sum(x_half)/Nsims
x_half_meansq = sum(x_half**2)/Nsims

x_end_mean = sum(x_end)/Nsims
x_end_meansq = sum(x_end**2)/Nsims

print(f'Halfway mean displacement: {x_half_mean}')
print(f'Halfway mean squared displacement: {x_half_meansq}')

print(f'End mean displacement: {x_end_mean}')
print(f'End mean squared displacement: {x_end_meansq}')

# Increasing k from 1-->4
Nsteps = 100
Nsims = 1000

x_half = np.zeros(Nsims)
x_end = np.zeros(Nsims)

for i in range(Nsims):
    traj = Hop_1D(0, 4, 0.1, Nsteps)
    x_half[i] = traj[round(Nsteps/2) + 1]
    x_end[i] = traj[Nsteps]

x_half_mean = sum(x_half)/Nsims
x_half_meansq = sum(x_half**2)/Nsims

x_end_mean = sum(x_end)/Nsims
x_end_meansq = sum(x_end**2)/Nsims

print(f'Halfway mean displacement: {x_half_mean}')
print(f'Halfway mean squared displacement: {x_half_meansq}')

print(f'End mean displacement: {x_end_mean}')
print(f'End mean squared displacement: {x_end_meansq}')

# Unequal probabilities: pleft > pright
def Hop_1D(x0, k, dt, Nsteps):
    pleft = 2*k*dt 
    pright = k*dt
    pstay = 1 - (pleft + pright)
    
    step_options = [-1, 0, 1]
    weights = [pleft, pstay, pright] 
    
    hop = np.zeros(Nsteps + 1)
    hop[0] = x0  
    for i in range(Nsteps):
        hop[i+1] = hop[i] + np.random.choice(step_options, p = weights)
        
    return hop

# Run new simulation
Nsteps = 100
Nsims = 1000

x_half = np.zeros(Nsims)
x_end = np.zeros(Nsims)

for i in range(Nsims):
    traj = Hop_1D(0, 1, 0.1, Nsteps)
    x_half[i] = traj[round(Nsteps/2) + 1]
    x_end[i] = traj[Nsteps]

x_half_mean = sum(x_half)/Nsims
x_half_meansq = sum(x_half**2)/Nsims

x_end_mean = sum(x_end)/Nsims
x_end_meansq = sum(x_end**2)/Nsims

print(f'Halfway mean displacement: {x_half_mean}')
print(f'Halfway mean squared displacement: {x_half_meansq}')

print(f'End mean displacement: {x_end_mean}')
print(f'End mean squared displacement: {x_end_meansq}')

'''1D Continuous Random Walk'''

def ContRandWalk1D(x0, t0, D, dt, Nsteps):
    
    times = np.zeros(Nsteps+1)
    times[0] = t0
    
    x = np.zeros(Nsteps+1)
    x[0] = x0
    
    for t in range(Nsteps):
        times[t+1] = times[t] + dt
        x[t+1] = x[t] + math.sqrt(2*D*dt)*np.random.randn(1)
    
    return times, x

times, x = ContRandWalk1D(0, 0, 100, 0.001, Nsteps)
fig, ax = plt.subplots(1,1)
ax.set_xlabel('Elapsed Time')
ax.set_ylabel('Position')
plt.scatter(times, x)
plt.show()

print(f'Halfway displacement: {x[50]} μm')
print(f'End displacement: {x[100]} μm')

'''2D Continuous Random Walk'''

def ContRandWalk2D(x0, y0, D, dt, Nsteps):
    sigma = math.sqrt(2*D*dt)
    
    times = np.zeros(Nsteps+1)
    x = np.zeros(Nsteps+1)
    y = np.zeros(Nsteps+1)
    
    times[0] = 0
    x[0] = x0
    y[0] = y0
    
    for t in range(Nsteps):
        times[t+1] = times[t] + dt
        x[t+1] = x[t] + sigma*np.random.randn(1)
        y[t+1] = y[t] + sigma*np.random.randn(1)
    
    return times, x, y

times, x, y = ContRandWalk2D(0, 0, 100, 0.001, 10000)

fig, ax = plt.subplots(1, 1, figsize=(14,7))
ax.set_title(f'2D Continuous Random Walk')
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.plot(x, y)
plt.show()

print(f'Halfway position: {x[50], y[50]}')
print(f'End position: {x[100], y[100]}')
print(f'Net distance from initial point: {math.sqrt(x[100]**2 + y[100]**2)} μm')

'''2D Continuous Random Walk with Harmonic Bond Bias'''

def ContRandWalk2D_Bias(x0, y0, D, T, dt, Nsteps):
    kT = 1
    sigma = math.sqrt(2*D*dt)
    
    times = np.zeros(Nsteps+1)
    x = np.zeros(Nsteps+1)
    y = np.zeros(Nsteps+1)
    
    times[0] = 0
    x[0] = x0
    y[0] = y0
    
    for t in range(Nsteps):
        times[t+1] = times[t] + dt
        x[t+1] = x[t] + sigma*np.random.randn(1) - dt*D/kT*8*x[t]
        y[t+1] = y[t] + sigma*np.random.randn(1) - dt*D/kT*2*y[t]
    
    return times, x, y

times, x, y = ContRandWalk2D_Bias(1, 1, 100, 300, 0.001, 10000)

fig, ax = plt.subplots(1, 1, figsize=(14,7))
ax.set_title(f'2D Continuous Random Walk with Bias: Harmonic Bond Potential V = 4x^2 + y^2')
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.plot(x, y)
plt.show()

print(f'Halfway position: {x[50], y[50]}')
print(f'End position: {x[100], y[100]}')
print(f'Net distance from initial point: {math.sqrt(x[100]**2 + y[100]**2)} μm')

