# NEUR472 2020
# Assignment 3 Q2 p2 Model answer
#
# MGP April 2020

# using Plots
# gr()
using PyPlot
using Distributions

"""
  Markov Neuron data type
"""
struct Markov_neuron

    # state vector
    x::Array{Float64,1} # [v,p] = [membrane potential, p_open]

    # parameters
    g::Float64    # channel conductance pS
    E::Float64    # equilibrium potential
    C::Float64    # capacitance pF
    N::Int64      # number of channels
    A::Float64

end

"""
   Neuron constructor for specified N and cell diameter in um
"""
function Markov_neuron(N::Int64, d::Float64)

    # specified parameters
    g_pS =  10.         # channel conductance pS
    E = -12.0        #  K+ equilibrium potential nb re RMP
    Cs =  1.0        # specific capacitance μF/cm^2
    p_init = 0.15    # initial K+ conductance (guess)

    # derived parameters
    A = π*d^2*1.0e-8       # membrane area in cm^2
    C = A*Cs               # capacitance in uF

    # overload default constructor
    return Markov_neuron([E, p_init], g, E, C, N, A)

end

"""
   Neuron constructor for specified membrane conductance in mS/cm^2
       and cell diameter in um
"""
function Markov_neuron(membrane_conductance::Float64, d::Float64)

    # specified parameters
    g_pS =  10.      # channel conductance in picosiemens
    E = -12.0        #  K+ equilibrium potential nb re RMP
    Cs =  1.0        # specific capacitance μF/cm^2
    p_init = 0.15    # initial K+ conductance (guess)

    # derived parameters
    A = π*d^2*1.0e-8      # membrane area in cm^2
    C = A*Cs              # capacitance in uF
    N = nchannels(g_pS,membrane_conductance,d)
    g_mS = g_pS*1.0e-6    # channel conductance in mS
                          # (equation dimensions: mv / ms = mS * mv / uF )

    # overload default constructor
    return Markov_neuron([E, p_init], g_mS, E, C, N, A)

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
    H-H α() and β() functions
"""
α(v) =  v == 10.0 ?  0.1 : 0.01*(10.0 - v)/(exp((10.0-v)/10)-1.0)
β(v) = 0.125*exp(-v/80.)

"""
   Neuron state update
   nb "neuron" is a reference to an object, it's fields will be updated
"""
function markov_update(neuron, I, Δt)

    # copy the state variables before you start messing with them!
    v = neuron.x[1]
    p = neuron.x[2]

    # coefficients of ODE for p  (τ dp/dt = p_inf - p)
    τ = 1.0/(α(v) + β(v))
    p_infinity = α(v)*τ

    n_open_distribution = Binomial(neuron.N, p)
    n_open = rand(n_open_distribution,1)[]

    #println(neuron.g*n_open/neuron.C)

    # nb p is state variable 2
    neuron.x[1] = v - Δt*(neuron.g*n_open*(v - neuron.E)-I)/neuron.C
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

Δt = .001                # simulation step length in ms nb consistent with τ
const T = 75.               # duration of simulation
const t = collect(0.0:Δt:T)  # simulation time array

pulseStart = 10.
pulseLen = 25.
pulseAmplitude = 100.
I = pulse(t, pulseStart, pulseLen, pulseAmplitude)

# for cell diameter 1cm: d = sqrt(1.0/pi)*1e4)
markov_neuron = Markov_neuron(36.,10.)   # construct a neuron

# burn in
for i in 1:100000
    markov_update(markov_neuron, 0.0, Δt)
end

x = fill(0.0, length(t), 2)  # output state array, 1 row per time step
x[1,:] = markov_neuron.x[:]         # initialize to neuron's state

for i in 2:length(t)

    #println(i)
    markov_update(markov_neuron, I[i], Δt)
    x[i,:] = markov_neuron.x[:] # copy membrane potential from neuron to v

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
