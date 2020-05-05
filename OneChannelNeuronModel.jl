# Simulation of 1-channel (type) neuron
# Hodgkin-Huxley, Markov model and Stochastic Langevin model
#
#   Markov model:      O<-->C
#   Parameters from Hodgkin and Huxley 1952

using PyPlot
using Distributions
using Printf

include("mgp_neuron_library.jl")

conductance = 36.  # mS/cm^2 (From Hodgkin & Huxley 1952)
Δt = .001

# cellDiam = sqrt(1.0/pi)*1e4 # cell diameter 1cm
# pulseAmplitude = 1.0e5 # 100μA

# cellDiam = 100. # μm
# pulseAmplitude = 100. # nA

cellDiam = 50. # μm
pulseAmplitude = 10. # nA

# cellDiam = 10. # μm
# pulseAmplitude = 0.5 # nA

# Weiner increments (W is Brownian motion)
dW = Normal(0.0,sqrt(Δt))


              # simulation step length in ms nb consistent with τ
const T = 30.               # duration of simulation
const t = collect(0.0:Δt:T)  # simulation time array

pulseStart = 10.
pulseLen = 10.

I = pulse(t, pulseStart, pulseLen, pulseAmplitude)

hneuron = HH_neuron(cellDiam)   # construct HH neuron
mneuron = Markov_neuron(36.,cellDiam)   # construct Markov neuron
sneuron = Stochastic_neuron(36.,cellDiam)   # construct Langevin neuron


# burn in
for i in 1:10000
    hh_update(hneuron, 0.0, Δt)
    markov_update(mneuron, 0.0, Δt)
    stochastic_update(sneuron, 0.0, Δt)
end

# create array to store state vector time series
hx = fill(0.0, length(t), length(hneuron.x))
hx[1,:] = hneuron.x[:]         # initialize to neuron's state

# Markov neuron
mx = fill(0.0, length(t), length(mneuron.x))
mx[1,:] = mneuron.x[:]  # initialize to neuron's state

# Langevin neuron
sx = fill(0.0, length(t), length(sneuron.x))  # output state array, 1 row per time step
sx[1,:] = sneuron.x[:]  # initialize to neuron's state

for i in 2:length(t)

    hh_update(hneuron, I[i], Δt)
    hx[i,:] = hneuron.x[:] # copy membrane potential from neuron to v

    markov_update(mneuron, I[i], Δt)
    mx[i,:] = mneuron.x[:] # copy membrane potential from neuron to v

    stochastic_update(sneuron, I[i], Δt)
    sx[i,:] = sneuron.x[:] # copy membrane potential from neuron to v


end



fig, (ax1, ax2, ax3, ax4,ax5) = subplots(nrows=5, ncols = 1, figsize = (10,8))
ax1.plot(t, sx[:,1])
ax1.plot(t, mx[:,1])
ax1.plot(t, hx[:,1])
ax1.legend(["Langevin", "Markov", "Hodgkin-Huxley"])
ax1.set_title(@sprintf(
  "Potassium Channel Model. Cell diameter = %.0fμm, %d channels.",
                        hneuron.d, mneuron.N))
ax1.set_ylabel("mV re RMP")
ax1.set_xlim(0,T)
ax1.set_xlabel("Membrane Potential")

ax2.plot(t, sx[:,2])
ax2.plot(t, mx[:,2])
ax2.plot(t, hx[:,2])
ax2.set_xlabel("Channel Open Probability")
ax2.set_ylabel("Pr")
ax2.set_xlim(0,T)
ax2.set_ylim(0.0, 0.5)

n = nchannels(10., 36., cellDiam)
ax3.plot(t, sx[:,3])
ax3.plot(t, mx[:,3])
ax3.plot(t, mneuron.N*hx[:,2])
ax3.set_xlabel("Number of Open Channels")
ax3.set_ylabel("Pr")
ax3.set_xlim(0,T)
ax3.set_ylim(0.0, n/2)

ax4.plot(t, sx[:,4])
ax4.plot(t, mx[:,4])
ax4.plot(t, hx[:,3])
ax4.set_xlabel("Channel Current")
ax4.set_ylabel("nA")
ax4.set_xlim(0,T)

ax5.plot(t, I)
ax5.set_xlabel("Injected Current       (time in ms)")
ax5.set_ylabel("nA")
ax5.set_xlim(0,T)

tight_layout()

display(fig)
# savefig("filename")     # save in cwd as .png
close(fig)  # otherwise fig stays in workspace
