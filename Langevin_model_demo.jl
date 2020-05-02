# NEUR472 2020
# Assignment 3 Q2 p2 Model answer
# Stochastic/Lamgevin neuron
#
# MGP April 2020

# using Plots
# gr()
using PyPlot
using Distributions

"""
  Stochastic Neuron data type
"""
struct Stochastic_neuron

    # state vector
    x::Array{Float64,1} # [v,p, i] = [potential, p_open, i_channel]

    # parameters
    g::Float64    # channel conductance pS
    E::Float64    # equilibrium potential
    C::Float64    # capacitance pF
    N::Int64      # number of channels
    d::Float64    # diameter μm
    A::Float64

end

# """
#    Neuron constructor for specified N and cell diameter in um
# """
# function Stochastic_neuron(N::Int64, d::Float64)
#
#     # specified parameters
#     g_pS =  10.         # channel conductance pS
#     E = -12.0        #  K+ equilibrium potential nb re RMP
#     Cs =  1.0        # specific capacitance μF/cm^2
#     p_init = 0.15    # initial K+ conductance (guess)
#
#     # derived parameters
#     A = π*d^2*1.0e-8       # membrane area in cm^2
#     C = A*Cs               # capacitance in uF
#
#     # overload default constructor
#     return Markov_neuron([E, p_init], g, E, C, N, A)
#
# end

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
    g_mS = g_pS*1.0e-6    # channel conductance in mS
                          # (equation dimensions: mv / ms = mS * mv / uF )

    # overload default constructor
    return Stochastic_neuron([E, p_init, i_init], g_mS, E, C, N, d, A)

end

"""
  Utility function for calculating mean and SD of Gaussian
      approximation to the binimial channel number
      in mS/cm^2 for a spherical cell of diameter d in microns.
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
   nb "neuron" is a reference to an object, its fields will be updated
"""
function stochastic_update(neuron, Inject, Δt)

    # copy the state variables before you start messing with them!
    # membrane potential and channel open probability
    # also improves readability by naming the state variables
    v = neuron.x[1]
    p = neuron.x[2]

    Inject = Inject*1.0e-3   # convert input current nA -> μA

    # coefficients of ODE for channel open probability, τ dp/dt = p_inf - p
    # time constant and equilibrium open probability
    τ = 1.0/(α(v) + β(v))
    p_infinity = α(v)*τ

    # generate noise using Normal approximation to binomial
    mean_noise = neuron.N*p
    sd_noise = sqrt(p*(1-p))  # nb Julia Normal(μ, σ)
    noisedist = Normal(0.0, sd_noise)
    noise = rand(noisedist,1)[]

    # channel current
    Ichannel = neuron.g*mean_noise*(v - neuron.E)

    # I specified in nA, convert to mA for dimensional correctness
    # Euler-Maruyama formula
    v = v - Δt*Ichannel/neuron.C +
            Δt*Inject/neuron.C - sqrt(Δt)*neuron.g*noise*(v - neuron.E)/neuron.C

    p = p + Δt*(p_infinity - p)/τ

    # numerical solution of ODE for p can overshoot range [0 1]
    # any tiny overshoot (p<0 or p>1) will crash rand()
    if (p>1.0) p = 1.0 end
    if (p<0.0) p = 0.0 end

    neuron.x[1] = v
    neuron.x[2] = p
    neuron.x[3] = Ichannel*1.0e3   # convert μA -> nA

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
const T = 30.               # duration of simulation
const t = collect(0.0:Δt:T)  # simulation time array

pulseStart = 10.
pulseLen = 10.
pulseAmplitude = 10.0   # nA
I = pulse(t, pulseStart, pulseLen, pulseAmplitude)

# for cell diameter 1cm: d = sqrt(1.0/pi)*1e4)
sneuron = Stochastic_neuron(36.,50.0)   # construct a neuron
println("Stochastic neuron with ", sneuron.N, " channels.")

# burn in
for i in 1:10000
    stochastic_update(sneuron, 0.0, Δt)
end

sx = fill(0.0, length(t), length(sneuron.x))  # output state array, 1 row per time step
sx[1,:] = sneuron.x[:]  # initialize to neuron's state

for i in 2:length(t)

    #println(i)
    stochastic_update(sneuron, I[i], Δt)
    sx[i,:] = sneuron.x[:] # copy membrane potential from neuron to v

end


# Plots
# nb useful to force y-axis limits for comparisons
fig, (ax1, ax2, ax3, ax4) = subplots(nrows=4, ncols = 1, figsize=(10,8))
ax1.plot(t, sx[:,1])
ax1.set_title("Langevin Model, "*string(sneuron.d)*
       "μm diameter cell with "*string(sneuron.N)*"  channels.")
ax1.set_ylabel("mV re RMP")
ax1.set_xlabel("Membrane Potential")
ax1.set_xlim(0, T)

ax2.plot(t, sx[:,2])
ax2.set_xlabel("Channel Open Probability")
ax2.set_ylabel("Pr")
ax2.set_ylim(0.0, 0.5)
ax2.set_xlim(0, T)

ax3.plot(t, sx[:,3])
ax3.set_xlabel("Channel Current")
ax3.set_ylabel("nA")
ax3.set_xlim(0, T)

ax4.plot(t, I)
ax4.set_xlabel("Injected Current       (time in ms)")
ax4.set_ylabel("nA")
tight_layout()

display(fig)
close(fig)  # otherwise fig stays in workspace
