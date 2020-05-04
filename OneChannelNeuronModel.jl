# Simulation of 1-channel (type) neuron
# Hodgkin-Huxley, Markov model and Stochastic Langevin model
#
#   Markov model:      O<-->C
#   Parameters from Hodgkin and Huxley 1952

using OneChannelNeuronModel_library


Δt = .01                # simulation step length in ms nb consistent with τ
const T = 30.               # duration of simulation
const t = collect(0.0:Δt:T)  # simulation time array

pulseStart = 10.
pulseLen = 10.
pulseAmplitude = 10. # nA
I = pulse(t, pulseStart, pulseLen, pulseAmplitude)

hneuron = HH_neuron(cellDiam)   # construct a neuron

# burn in
for i in 1:10000
    hh_update(hneuron, 0.0, Δt)
end

# create array to store state vector time series
hx = fill(0.0, length(t), length(hneuron.x))
hx[1,:] = hneuron.x[:]         # initialize to neuron's state

for i in 2:length(t)

    hh_update(hneuron, I[i], Δt)
    hx[i,:] = hneuron.x[:] # copy membrane potential from neuron to v

end

# plot(t, hcat(x, I),
#     layout = (3,1),  label = :none,
#     title = ["membrane potential" "channel open probability" "input current"])

fig, (ax1, ax2, ax3, ax4,ax5) = subplots(nrows=5, ncols = 1, figsize = (10,8))
ax1.plot(t, hx[:,1])
ax1.set_title("Hodgkin-Huxley Model, "*string(hneuron.d)*"μm diameter cell")
ax1.set_ylabel("mV re RMP")
ax1.set_xlim(0,T)
ax1.set_xlabel("Membrane Potential")

ax2.plot(t, hx[:,2])
ax2.set_xlabel("Channel Open Probability")
ax2.set_ylabel("Pr")
ax2.set_xlim(0,T)
ax2.set_ylim(0.0, 0.5)

n = nchannels(10., 36., cellDiam)
ax3.plot(t, n.*hx[:,2])
ax3.set_xlabel("Number of Open Channels")
ax3.set_ylabel("Pr")
ax3.set_xlim(0,T)
ax3.set_ylim(0.0, n/2)

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
close(fig)  # otherwise fig stays in workspace


println("hello")
