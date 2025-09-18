using ITensors, ITensorMPS, HDF5

### Initial Parameters ###
N = parse(Int, ARGS[1]) #Number of lattice sites per dimension
d = 1 #Number of spatial dimensions
dx = 1 #Lattice spacing

n_0 = round(Int, ((N-1)/2)) #Index of the point at the center of the lattice.
dp = 4pi/(N*dx) #0.1 #Discrete change in momentum

Delta = 0 #A constant factor that shifts the values of the momentum lattice. Delta=0 corresponds to periodic boundary conditions, while Delta=1/2 corresponds to "twisted" boundary conditions
mass = 0
p_i = [-(n_0*dp)+(s-Delta)*dp for s in 0:N-1] #List containing the momentum at each site in the lattice
om_i = [sqrt(((2/dx)*sin(p_i[i]*(dx/2)))^2 + mass^2) for i in 1:N] #Angular frequency of the field

### Hamiltonian ###
H = OpSum() #Evaluate the sum of the pi^2 and phi operators that make up the Hamiltonian
for i in 1:N
    global H += (om_i[i]*dp)/(2pi), "a", i, "a†", i
    global H += (om_i[i]*dp)/(2pi)*(1/2)*(2pi/dp), "Id", i
end

sites = siteinds("Boson", N; dim=9) #Create ITensor "SFT" sites

### Create Vacuum State ###
psi_ansatz = random_mps(sites;linkdims=4)
energy, psi_gs = dmrg(MPO(H, sites), psi_ansatz; nsweeps=(n_0+1), maxdim=200, cutoff=1e-8);

### Define the Field Operator and Wilson Line operators ###
function phi_n(m::Real) #Field value in the lightlike direction, i.e. phi_x_{n_0+m}
    phi_lightlike = OpSum()
    for s in 0:N-1
        i = s+1
        phi_lightlike += (1/sqrt(N))*(1/sqrt(2*om_i[i]))*exp(-im*(p_i[i]*m*dx - om_i[i]*m*dx)), "a", (i) #n*m*dx = (1,1)*m*dx = (m*dx,m*dx)
        phi_lightlike += (1/sqrt(N))*(1/sqrt(2*om_i[i]))*exp(im*(p_i[i]*m*dx - om_i[i]*m*dx)), "a†", (i)
    end
    return phi_lightlike
end

function phi_nbar(m::Real) #Field value in the anti-lightlike direction, i.e. phi_x_{n_0-m}
    phi_antilightlike = OpSum()
    for s in 0:N-1
        i = s+1
        phi_antilightlike += (1/sqrt(N))*(1/sqrt(2*om_i[i]))*exp(im*(p_i[i]*m*dx + om_i[i]*m*dx)), "a", (i) #nbar*m*dx = (1,-1)*m*dx = (m*dx,-m*dx)
        phi_antilightlike += (1/sqrt(N))*(1/sqrt(2*om_i[i]))*exp(-im*(p_i[i]*m*dx + om_i[i]*m*dx)), "a†", (i)
    end
    return phi_antilightlike
end

function Y_n(g::Real)
    Y_n_op = OpSum()
    for m in 0:n_0
        Y_n_op += im*g*dx*phi_n(m)
    end
    return Y_n_op
end

function Y_nbar_dag(g::Real)
    Y_nbar_dag_op = OpSum()
    for m in 0:n_0
        Y_nbar_dag_op += -im*g*dx*phi_nbar(m)
    end
    return Y_nbar_dag_op
end

function YnbdYn(g::Real)
    return MPO(Y_nbar_dag(g)+Y_n(g), sites)
end

### Prepare for Data Collection ###
t_range = range(0, stop=40, length=201) #Range of the timesteps
g_list = [0.5] #List of coupling constant values
cutoff_list = [1e-27] #List of TDVP cutoff parameters for each value of g
if N > 14 #If N > 14, increase the threshold for ITensor's contraction warning
    ITensors.set_warn_order(N)
    ITensors.set_warn_order(N)
else #Else, reset the warning threshold to the default number of contractions, 14
    ITensors.set_warn_order(14)
    ITensors.set_warn_order(14)
end


### Calculate S(t) ###
#Create the states |Psi_0> = Y_nbar^dag(0)Y_n(0)|Ω> and |Psi_t> = e^{iHt}Y_nbar^dag(0)Y_n(0)e^{-iHt}|Ω> for some value of g and t
function MakeStates(g, t, cutoff)
    Psi_0 = tdvp(YnbdYn(g), 1, psi_gs; nsteps=1, maxdim=200, cutoff=cutoff, normalize=true) #tdvp calculates exp(op*t)|psi>. By setting t=1, tdvp reduces to exp(op)|psi>
    
    psi_t = tdvp(MPO(-im*H, sites), t, psi_gs; nsteps=1, maxdim=200, cutoff=cutoff, normalize=true) #Time evolve the ground state
    psi_t = tdvp(YnbdYn(g), 1, psi_t; nsteps=1, maxdim=200, cutoff=cutoff, normalize=true) #Apply the Wilson line operator
    Psi_t = tdvp(MPO(im*H, sites), t, psi_t; nsteps=1, maxdim=200, cutoff=cutoff, normalize=true) #Time evolve back

    return Psi_0, Psi_t
end

for (i, g) in enumerate(g_list)
    S_t_list = ComplexF64[] #Create temporary lists for storing S(t), the Psi_0 MPS, and the Psi_t MPS for a given value of g
    SF_data = h5open("SF_data/N=$N,g=$g,m=$mass,dp=$dp", "w")
    SF_MPS = h5open("SF_MPS/N=$N,g=$g,m=$mass,dp=$dp", "w")
    for t in t_range
        #Calculate S(t) by taking the inner product between |Psi_0> and |Psi_t>
        Psi_0, Psi_t = MakeStates(g, t, cutoff_list[i])

        if t == t_range[end]
            SF_MPS["Psi_t, t=$t"] = Psi_t #Create a dataset containing the Psi_t MPS at timestep t for a given value of g
        end
    
        S_t = inner(Psi_t, Psi_0)
        push!(S_t_list, S_t) #Store the S(t) value at timestep t
    end
    SF_data["S(t)"] = S_t_list #Create a dataset containing the list of S(t) values at each timestep t for a given value of g
    close(SF_data)
    close(SF_MPS)
end

#Store the list of S(t) values into a dictionary 'keyed' by the corresponding value of g
S_t_list_g = Dict()
for g in g_list
    SF_data = h5open("SF_data/N=$N,g=$g,m=$mass,dp=$dp", "r")
    S_t_list_g["g=$g"] = [SF_data["S(t)"][i] for i in 1:length(t_range)]
    close(SF_data)
end

### Calculate S(E) ###
function S(E::Real, S_t)
    dt = (2/dx)/(length(t_range)-1)
    FTSoftFunc = 0
    for t in 1:(length(t_range)-1)
        f1 = exp(im*E*t_range[t])*S_t[t]
        f2 = exp(im*E*t_range[t+1])*S_t[t+1]
        FTSoftFunc += (dt/2)*(f1+f2)
    end
    return FTSoftFunc
end

E_range = 0:0.05:10
for (i, g) in enumerate(g_list)
    S_E_list = ComplexF64[] #Create a temporary list for containing S(E) values for a given value of g
    S_E_list = [S(E, S_t_list_g["g=$g"]) for E in E_range] #Store S(E) for each value of E
    SF_data = h5open("SF_data/N=$N,g=$g,m=$mass,dp=$dp", "r+")
    SF_data["S(E)"] = S_E_list #Create a dataset containing the list of S(E) values at each energy E for a given value of g
    close(SF_data)
end