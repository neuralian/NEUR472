# NEUR472 2020
# Assignment 3 Q2 Model answer
#
# MGP April 2020

# using Plots
# gr()
using PyPlot

"""
  Neuron data type
"""
struct HH_neuron

    # state vector
    x::Array{Float64,1} # [v,p] = [membrane potential, open probability]

    # parameters
    g::Float64    # maximal conductance mS/cm^2
    E::Float64    # equilibrium potential mV
    C::Float64    # capacitance μF/cm^2

end

"""
   HH Neuron constructor
"""
function HH_neuron()

    g =  36.0       # K+ conductance mS/cm^2
    E = -12.0       #  K+ equilibrium potential nb re RMP
    C =  1.0         # capacitance μF/cm^2
    p_init = 0.15    # initial K+ conductance (guess)

    # overload default constructor
    return HH_neuron([E, p_init], g, E, C )

end


"""
    H-H α() and β() functions
"""
α(v) =  v == 10.0 ?  0.1 : 0.01*(10.0 - v)/(exp((10.0-v)/10)-1.0)
β(v) = 0.125*exp(-v/80.)

"""
   HH Neuron state update
   nb "neuron" is a reference to an object, it's fields will be updated
"""
function hh_update(neuron, I, Δt)

    # copy the state variables before you start messing with them!
    v = neuron.x[1]
    p = neuron.x[2]

    # coefficients of ODE for p  (τ dp/dt = p_inf - p)
    τ = 1.0/(α(v) + β(v))
    p_infinity = α(v)*τ

    # nb p is state variable 2
    neuron.x[1] = v - Δt*(neuron.g*p*(v - neuron.E)-I)/neuron.C
    neuron.x[2] = p + Δt*(p_infinity - p)/τ

end

"""
   pulse waveform same length as t
"""
function pulse(t, start, len, amplitude)

u = zeros(length(t))
  u[findall( t-> (t>=start) & ( t<start+len), t)] .= amplitude

  return u
end

Δt = .01                # simulation step length in ms nb consistent with τ
const T = 75.               # duration of simulation
const t = collect(0.0:Δt:T)  # simulation time array

pulseStart = 10.
pulseLen = 25.
pulseAmplitude = 100.0
I = pulse(t, pulseStart, pulseLen, pulseAmplitude)

hh_neuron = HH_neuron()   # construct a neuron

# burn in
for i in 1:10000
    hh_update(hh_neuron, 0.0, Δt)
end

x = fill(0.0, length(t), 2)  # output state array, 1 row per time step
x[1,:] = hh_neuron.x[:]         # initialize to neuron's state

for i in 2:length(t)

    hh_update(hh_neuron, I[i], Δt)
    x[i,:] = hh_neuron.x[:] # copy membrane potential from neuron to v

end

# plot(t, hcat(x, I),
#     layout = (3,1),  label = :none,
#     title = ["membrane potential" "channel open probability" "input current"])

fig, (ax1, ax2, ax3) = subplots(nrows=3, ncols = 1)
ax1.plot(t, x[:,1])
ax1.set_title("Membrane Potential")
ax1.set_ylabel("mV re RMP")
ax2.plot(t, x[:,2])
ax2.set_title("Channel Open Probability")
ax2.set_ylabel("Pr")
ax3.plot(t, I)
ax3.set_title("Injected Current")
ax3.set_ylabel("uA")
ax3.set_xlabel("time /ms")
tight_layout()

display(fig)
