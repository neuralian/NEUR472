# Simulation of 1-channel (type) neuron
# Hodgkin-Huxley, Markov model and Stochastic Langevin model
#
#   Markov model:      O<-->C
#   Parameters from Hodgkin and Huxley 1952

using PyPlot
using Distributions
using Printf

conductance = 36.  # mS/cm^2 (From Hodgkin & Huxley 1952)
Δt = .001

cellDiam = sqrt(1.0/pi)*1e4 # cell diameter 1cm
pulseAmplitude = 1.0e5 # 100μA

# cellDiam = 100. # μm
# pulseAmplitude = 100. # nA

cellDiam = 50. # μm
pulseAmplitude = 10. # nA

cellDiam = 10. # μm
pulseAmplitude = 0.5 # nA

# Weiner increments (W is Brownian motion)
dW = Normal(0.0,sqrt(Δt))


"""
  HH-Neuron data type
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
   Markov Neuron constructor for specified N and cell diameter in um
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
   Markov Neuron constructor for specified membrane conductance in mS/cm^2
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
    g_mS = g_pS*1.0e-6    # channel conductance in mS
                          # (equation dimensions: mv / ms = mS * mv / uF )

    # overload default constructor
    return Markov_neuron([E, p_init, n_init, i_init], g_mS, E, C, N, d, A)

end

"""
   Markov Neuron state update
   nb "neuron" is a reference to an object, it's fields will be updated
"""
function markov_update(neuron, Inject, Δt)

    # copy the state variables before you start messing with them!
    v = neuron.x[1]
    p_open = neuron.x[2]
    n_open = neuron.x[3]

    # coefficients of ODE for p  (τ dp/dt = p_inf - p)
    τ = 1.0/(α(v) + β(v))
    p_infinity = α(v)*τ

    Inject = Inject*1.0e-3  # nA -> μA

    # number of channels that open
    # opening probability is α(v)Δt
    # Binomial N must be integer
    n_opening_distribution = Binomial(Int64(neuron.N-n_open), α(v)*Δt)
    n_opening = rand(n_opening_distribution,1)[]

    # number of channels that close
    # closing probability is β(v)Δt
    n_closing_distribution = Binomial(Int64(n_open), β(v)*Δt)
    n_closing = rand(n_closing_distribution,1)[]

    # update number of open channels
    n_open = n_open + n_opening - n_closing

    # current through channels
    Ichannel = neuron.g*n_open*(v - neuron.E)

    # external current I in nA, convert to mA for dimensional correctness
    v = v - Δt*(Ichannel -Inject)/neuron.C
    p_open = p_open + Δt*(p_infinity - p_open)/τ

    # # numerical solution of ODE for p can overshoot range [0 1]
    # # any tiny overshoot (p<0 or p>1) will crash rand()
    # if (p>1.0) p = 1.0 end
    # if (p<0.0) p = 0.0 end
    # neuron.x[2] = p

    # update the state vector
    neuron.x[1] = v
    neuron.x[2] = p_open
    neuron.x[3] = n_open
    neuron.x[4] = Ichannel*1.0e3  # μA -> nA

end

"""
  Stochastic Langevin Neuron data type
"""
struct Stochastic_neuron

    # state vector
    x::Array{Float64,1} # [v,p,n, i] = [potential, p_open, n_open, i_channel]

    # parameters
    g::Float64    # channel conductance μS
    E::Float64    # equilibrium potential
    C::Float64    # capacitance pF
    N::Int64      # number of channels
    d::Float64    # diameter μm
    A::Float64

end


"""
   Neuron constructor for specified membrane conductance in mS/cm^2
       and cell diameter in um
"""
function Stochastic_neuron(membrane_conductance::Float64, d::Float64)

    # specified parameters
    g_pS =  10.      # channel conductance in picosiemens
    E = -12.0        #  equilibrium potential nb re RMP
    Cs =  1.0        # specific capacitance μF/cm^2
    p_init = 0.15    # initial conductance (guess)
    i_init = 0.0     # initial current

    # derived parameters
    A = π*d^2*1.0e-8      # membrane area in cm^2
    C = A*Cs              # capacitance in uF
    N = nchannels(g_pS,membrane_conductance,d)
    g_uS = g_pS*1.0e-6    # channel conductance in μS
                          # (equation dimensions: mv / ms = mS * mv / uF )

    # overload default constructor
    return Stochastic_neuron([E, p_init, p_init*N, i_init], g_uS, E, C, N, d, A)

end

"""
   Stochastic Langevin Neuron state update
   nb "neuron" is a reference to an object, its fields will be updated
"""
function stochastic_update(neuron, Inject, Δt)

    # copy the state variables before you start messing with them!
    # membrane potential and channel open probability
    # also improves readability by naming the state variables
    v = neuron.x[1]
    p = neuron.x[2]
    n_open = neuron.x[3]

    Inject = Inject*1.0e-3   # convert input current nA -> μA

    # coefficients of stochastic ODE for channel open probability,
    # τ dp/dt = p_inf - p + sqrt(γ)dW
    # time constant and equilibrium open probability
    a = α(v)
    b = β(v)
    τ = 1.0/(a+b)
    p_infinity = a*τ
    γ = (a*(1-p) + b*p)/neuron.N

    Ichannel = neuron.g*n_open*(v - neuron.E)

    # Euler integration for membrane potential
    v = v - Δt*(Ichannel - Inject)/neuron.C

    # Euler-Maruyama integration for open prob
    p = p + Δt*(p_infinity - p)/τ + sqrt(γ)*rand(dW,1)[]

    # numerical solution of ODE for p can overshoot range [0 1]
    # any tiny overshoot (p<0 or p>1) will crash rand()
    if (p>1.0) p = 1.0 end
    if (p<0.0) p = 0.0 end

    neuron.x[1] = v
    neuron.x[2] = p
    neuron.x[3] = neuron.N*p
    neuron.x[4] = Ichannel*1.0e3   # convert μA -> nA

end




"""
    H-H α() and β() functions
"""
α(v) =  v == 10.0 ?  0.1 : 0.01*(10.0 - v)/(exp((10.0-v)/10)-1.0)
β(v) = 0.125*exp(-v/80.)






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

function JohnsonNyquist(n::Int64, r::Float64, f::Float64)

   kB = 1.38e-23 # Boltzmann constant
   T = 300.  # Kelvin temperature
   JNnoise = Normal(0.0, sqrt(4.0*kB*T*r*f))
   w = rand(JNnoise, n)*1.0e3

end


"""
   pulse waveform same length as t
"""
function pulse(t, start, len, amplitude)

u = zeros(length(t))
  u[findall( t-> (t>=start) & ( t<start+len), t)] .= amplitude

  return u
end


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

# plot(t, hcat(x, I),
#     layout = (3,1),  label = :none,
#     title = ["membrane potential" "channel open probability" "input current"])

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
close(fig)  # otherwise fig stays in workspace
