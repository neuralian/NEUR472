using PyPlot

Δt = 0.01                    # ms
const T = 30.0
const t = collect(0.0:Δt:T)
pulseStart = 10.            # ms
pulseLen = 10.              # ms
PulseAmpl = 10.             # nA

struct Neuron
    # state vector
    x::Array{Float64,1} # [v, p, i]
    y::Array{Float64,1}

    # parameters
    g::Float64  # conductance            mS/cm^2
    E::Float64  # equilibrium potential  mV
    C::Float64  # Capacitance            μF/cm^2
    d::Float64  # diameter               μm
    A::Float64  # Membrane area          cm^2

end

"""
    Neuron Constructor
    Specify diameter in μm
    Other parameters inserted from Hodgkin & Huxley 1952
"""
function Neuron(d::Float64)

    gs = 36.0   # specific conductance mS/cm^2
    E = -12.0  # K+ equilibrium  mV
    Cs = 1.0    # specific capacitance μF/cm^2

    #
    A = π*d^2*1.0e-8  # membrane area in cm^2
    C = Cs*A          # neuron capacitance
    g = gs*A          # conductance
    v0 = 0.0          # initial voltage RMP
    p0 = 0.15         # initial open state prob
    i0 = 0.0           # initial current
    x = [v0, p0, i0]   # initial state
    y = [0.0]

  Neuron(x, y, g, E, C, d, A)

end

"""
  H-H rate parameters
"""
α(v) = v == 10.0 ?  0.1 : 0.01*(10.0-v)/(exp((10.0-v)/10.0)-1.0)
β(v) = 0.125*exp(-v/80.)


"""
    H-H state update
"""
function update(neuron, Inject, Δt)

    # copy the state variables
    v = neuron.x[1]
    p = neuron.x[2]

    Inject = Inject*1.0e-3  # convert nA -> μA

    # coeffs of ODE for channel kinetics
    # τ dp/dt = p_inf - p
    τ = 1.0/(α(v) + β(v))
    p_infinity = α(v)*τ

    Ichannel = neuron.g*p*(v-neuron.E)

    # update state
    neuron.x[1] = v - Δt*(Ichannel-Inject)/neuron.C
    neuron.x[2] = p + Δt*(p_infinity - p)/τ
    neuron.x[3] = Ichannel*1.0e3    # convert μA -> nA

end


"""
  Pulse generator
"""
function pulse(t, start, len, amplitude)

    u = zeros(length(t))
    u[findall( t-> (t>=start) & (t<start+len), t)] .= amplitude

    return u
end

neuron = Neuron(50.)

# burn in
for i in 1:10000

    update(neuron, 0.0, Δt)

end

# input current
Inject = pulse(t, pulseStart, pulseLen, PulseAmpl)

# array to hold state
x = fill(0.0, length(t), length(neuron.x))
x[1,:] = neuron.x[:]

for i = 2:length(t)

    update(neuron, Inject[i], Δt)
    x[i, :] = neuron.x[:]

end

fig, (ax1, ax2) = subplots(nrows=2, ncols=1, figsize=(10,4))
ax1.plot(t, x[:,1])
ax1.set_title("Hodgkin_Huxley Model")
ax1.set_ylabel("mV re RMP")
ax1.set_xlim(0.0, T)
ax1.set_xlabel("Membrane Potential")


tight_layout()
display(fig)
savefig("HHmodel") # save as png
close(fig)
