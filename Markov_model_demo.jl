# NEUR472 2020
# Assignment 3 Q2 p3 Model answer
# Markov neuron
#
# MGP April 2020

# using Plots
# gr()
using PyPlot
using Distributions

cellDiam = 5.
Δt = .01                # simulation step length in ms nb consistent with τ
pulseAmplitude = .5   # nA

"""
  Markov Neuron data type
"""
struct Markov_neuron

    # state vector
    x::Array{Float64,1} # [v,p, n, i] = [potential, p_open, n_open, current]

    # parameters
    g::Float64    # channel conductance pS
    E::Float64    # equilibrium potential
    C::Float64    # capacitance pF
    N::Int64      # number of channels
    d::Float64    # cell diameter in μm
    A::Float64    #  membrane area in cm^2

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
    n_init = round(N*p_init) # initial n open (must be convertable to Int64)

    # derived parameters
    A = π*d^2*1.0e-8       # membrane area in cm^2
    C = A*Cs               # capacitance in uF

    # overload default constructor
    return Markov_neuron([E, p_init, n_init], g, E, C, N, d, A)

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
    i_init = 0.0     # initial current


    # derived parameters
    A = π*d^2*1.0e-8      # membrane area in cm^2
    C = A*Cs              # capacitance in uF
    N = nchannels(g_pS,membrane_conductance,d)  # total channels
    n_init = round(N*p_init) # initial n open
    g_mS = g_pS*1.0e-9    # channel conductance in mS
                          # (equation dimensions: mv / ms = mS * mv / uF )

    # overload default constructor
    return Markov_neuron([E, p_init, n_init, i_init], g_mS, E, C, N, d, A)

end

"""
  Utility function for calculating number of channels with a given conductance
    in pS required to get specified conductance in mS/cm^2 for a spherical
    cell of diameter d in microns.
"""
function nchannels(s_channel, s_membrane, celldiam)

   A = π*celldiam^2*1.0e-8            # membrane area in cm^2
   mS = A*s_membrane                  # capacitance in mS
   pS = 1.0e9*mS                      # capacitance in pS
   n = Int64(round(pS/s_channel))     # number of channels

end

""""
    Johnson-Nyquist Noise in mV
    Function of electrode resistance and bandwidth
    Returns column vector of n samples
"""

function JohnsonNyquist(n::Int64, r::Float64, f::Float64)

   kB = 1.38e-23 # Boltzmann constant
   T = 300.  # Kelvin temperature
   JNnoise = Normal(0.0, sqrt(4.0*kB*T*r*f))
   w = rand(JNnoise, n)*1.0e3

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
function markov_update(neuron, Inject, Δt)

    # copy the state variables before you start messing with them!
    v = neuron.x[1]
    p = neuron.x[2]
    n = neuron.x[3]

    Inject = Inject*1.0e-3  # nA -> μA

    # current through channels
    Ichannel = neuron.g*n*(v - neuron.E)

    # updated membrane potential
    v = v - Δt*(Ichannel -Inject)/neuron.C

    # coefficients of ODE for channel state prob  (τ dp/dt = p_inf - p)
    τ = 1.0/(α(v) + β(v))
    p_infinity = α(v)*τ

    # change in channel state probability (difference approx to ODE)
    Δp = Δt*(p_infinity - p)/τ

    # updated channel state probability
    p = p + Δp

    # constrain probability to (0,1)
    if (p>1.0) p = 1.0 end
    if (p<0.0) p = 0.0 end

    # distribution of number of channels that open (can be <0)
    n_opening_dist = Binomial(Int64(neuron.N-n), abs(Δp))   

    # number of channels that open [close]
    n_opening = rand(n_opening_dist,1)[]

    # update number of open channels
    n = n + sign(Δp)*n_opening

    # update the state vector
    neuron.x[1] = v
    neuron.x[2] = p
    neuron.x[3] = n #rand(n_dist, 1)[]
    neuron.x[4] = Ichannel*1.0e3  # μA -> nA

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
I = pulse(t, pulseStart, pulseLen, pulseAmplitude)

# for cell diameter 1cm: d = sqrt(1.0/pi)*1e4)
mneuron = Markov_neuron(36.,cellDiam)   # construct a neuron

# burn in
for i in 1:10000
    markov_update(mneuron, 0.0, Δt)
end

# create array to hold state vector time series, 1 row per time step
mx = fill(0.0, length(t), length(mneuron.x))
mx[1,:] = mneuron.x[:]  # initialize to neuron's state

for i in 2:length(t)

    #println(i)
    markov_update(mneuron, I[i], Δt)
    mx[i,:] = mneuron.x[:] # copy membrane potential from neuron to v

end

# plot(t, hcat(x, I),
#     layout = (3,1),  label = :none,
#     title = ["membrane potential" "channel open probability" "input current"])

fig, (ax1, ax2, ax3, ax4, ax5) =subplots(nrows=5, ncols = 1, figsize = (10, 10))
ax1.plot(t, mx[:,1] + JohnsonNyquist(size(mx,1), 2.0e6, 1.0e5))
ax1.set_title("Markov Model, "*string(mneuron.d)*
       "μm diameter cell with "*string(mneuron.N)*"  channels.")
ax1.set_ylabel("mV re RMP")
ax1.set_xlabel("Membrane Potential re RMP")
ax1.set_xlim(0.0, T)

ax2.plot(t, mx[:,2])
ax2.set_xlabel("Open Probability")
ax2.set_ylabel("Pr")
ax2.set_xlim(0.0, T)
ax2.set_ylim(0.0, 0.5)

ax3.plot(t, mx[:,3])
ax3.set_xlabel("Number of Open Channels")
ax3.set_ylabel("n")
ax3.set_xlim(0.0, T)

ax4.plot(t, mx[:,4])
ax4.set_ylabel("μA")
ax4.set_xlabel("Channel Current")
ax4.set_xlim(0.0, T)

ax5.plot(t, I)
ax5.set_ylabel("μA")
ax5.set_xlabel("Injected Current         (time in ms)")
ax5.set_xlim(0.0, T)
tight_layout()

display(fig)
close(fig)
