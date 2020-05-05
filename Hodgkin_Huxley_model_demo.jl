# NEUR472 2020
# Assignment 3 Q2 Model answer
#
# MGP April 2020

# using Plots
# gr()
using PyPlot

cellDiam = 50.0
Δt = .0001              # simulation step length in ms nb consistent with τ




















































"""
  Neuron data type
"""
struct HH_neuron

    # state vector
    x::Array{Float64,1} # [v,p, i] = [potential, p_open, i_channel]

    # parameters
    g::Float64    # maximal conductance mS/cm^2
    E::Float64    # equilibrium potential mV
    C::Float64    # capacitance μF/cm^2
    d::Float64    # cell diameter in μm
    A::Float64    # membrane area in cm^2

end

"""
   HH Neuron constructor
"""
function HH_neuron(d::Float64)

    p_init = 0.15    # initial K+ conductance (guess)
    i_init = 0.0     # initial current

    gs =  36.0        # specific conductance mS/cm^2
    E = -12.0        #  K+ equilibrium potential nb re RMP
    Cs =  1.0        # specific capacitance μF/cm^2

    A = π*d^2*1.0e-8 # membrane area cm^2
    C = Cs*A         # membrane capacitance
    g = gs*A         # membrane conductance

    # overload default constructor
    return HH_neuron([E, p_init, i_init], g, E, C, d, A )

end


"""
    H-H α() and β() functions
"""
α(v) =  v == 10.0 ?  0.1 : 0.01*(10.0 - v)/(exp((10.0-v)/10)-1.0)
β(v) = 0.125*exp(-v/80.)

"""
   HH Neuron state update
   nb "neuron" is a reference to an object, its fields will be updated
"""
function hh_update(neuron, Inject, Δt)

    # copy the state variables before you start messing with them!
    v = neuron.x[1]
    p = neuron.x[2]

    Inject = Inject*1.0e-3  # convert nA -> μA

    # coefficients of ODE for channel open probability, τ dp/dt = p_inf - p
    τ = 1.0/(α(v) + β(v))
    p_infinity = α(v)*τ

    # channel current
    Ichannel = neuron.g*p*(v - neuron.E)

    # update state
    # nb input I is specified in nA, convert to mA for dimensional correctness
    neuron.x[1] = v - Δt*(Ichannel-Inject)/neuron.C
    neuron.x[2] = p + Δt*(p_infinity - p)/τ
    neuron.x[3] = Ichannel*1.0e3  # convert μA -> nA

end

"""
  Utility function for calculating number of channels with a given conductance
    in pS required to get specified conductance in mS/cm^2 for a spherical
    cell of diameter d in microns.
"""
function nchannels(s_channel, s_membrane, celldiam)

   A = π*celldiam^2*1.0e-8            # membrane area in cm^2
   mS = A*s_membrane                  # capacitance in mS
   pS = 1.0e6*mS                      # capacitance in pS
   n = Int64(round(pS/s_channel))     # number of channels

end

"""
   pulse waveform same length as t
"""
function pulse(t, start, len, amplitude)

u = zeros(length(t))
  u[findall( t-> (t>=start) & ( t<start+len), t)] .= amplitude

  return u
end


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
