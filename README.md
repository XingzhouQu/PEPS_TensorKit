## Square lattice Fermionic/Bosonic iPEPS code based on TensorKit.jl.

The fermionic sign is attracted through swap gates.

We provide:
1. Simple Update (Nearest-Neghbor / Next-Nearest-Neighbor interations)
1. Fast Full Update (Nearest-Neghbor interactions)
1. CTMRG
1. Observables (1-site and 2-site operators)

One can customize the LocalSpace following the style of [FiniteMPS.jl](https://github.com/Qiaoyi-Li/FiniteMPS.jl)

***

### Basic usage
The bosonic and fermionic algorithms are implemented independently.
Since the code have not been registed as a package, one needs to explicitly `include` the corresponding files before any calculation.
For example, in "example/Nick_2orbtJ.jl" we show the fermionic iPEPS code to calculate the ground state of bilayer Nickelate superconductor La$_3$Ni$_2$O$_7$,
based on a bilayer two-orbital t-J model.

    include("../iPEPS_Fermionic/iPEPS.jl")  # This is a fermionic system so we choose the fermionic iPEPS.
    include("../CTMRG_Fermionic/CTMRG.jl")  # include the fermionic CTMRG algorithm
    include("../models/tJ_Z2SU2_2orb.jl")  # In this file we define the local Hilbert space of bilayer La$_3$Ni$_2$O$_7$, which is a composite space of four t-J sites (d=81) with Z2charge × SU(2)spin symmetry .
    include("../simple_update_Fermionic/simple_update_anisotropic.jl")  # import the simple update algorithm. In the Hamiltonian, interorbital hybridization terms exhibit ± signs in x/y direction, so we use the anisotropic version.
    include("../Cal_Obs_Fermionic/Cal_Obs.jl")  # Calculate the on-site and nearest-neighbor observables, especially the interlayer pairing, which becomes on-site operators in our setup.

After that we initialize the parameters by convention:

    para = Dict{Symbol,Any}()
    para[:property] = value

The simple update begins with a Γ-λ form iPEPS (entanglement spectrums on bond), randomly initialized as:

    ipepsγλ = iPEPSΓΛ(pspace, aspacelr, aspacetb, Lx, Ly; dtype=Float64)

where `pspace` is the physical space, `aspacelr` and `aspacetb` are the auxiliary spaces for the left-right and top-bottom indices.

Then the simple update is performed by:

    simple_update_aniso!(ipepsγλ, tJ2orb_hij, para)

where the Hamiltonian `tJ2orb_hij::Function` is defined in the model file `/models/tJ_Z2SU2_2orb.jl`.

After the convergence of simple update, we recover the normal form of iPEPS, get its Hermition conjugate `ipepsbar` and initialize the environment tensors:

    ipeps = iPEPS(ipepsγλ)
    ipepsbar = bar(ipeps)
    envs = iPEPSenv(ipeps)

Then calculate the environment tensors by CTMRG:

    CTMRG!(ipeps, ipepsbar, envs, para[:χ], para[:CTMit]; parallel=para[:CTMparallel], threshold=para[:CTMthreshold])

where `para[:CTMparallel]::Bool` controls whether the CTMRG is parallelized or not. 

`para[:CTMparallel]==True` means the left-right environment are calculated at the same time and top-bottom environments are calculated at the same time.

`para[:CTMparallel]==False` means the environments at four dierctions are calculated one by one.

Finally we use the `Cal_Obs_1site` and `Cal_Obs_2site` functions to calculate the observables. I am planning to improve this part to support the calculations of long-range correlations.

***

### Tips
The function `save` of JLD2.jl are reloaded in file "iPEPS_Fermionic(Bosonic)/util.jl" so one can save and load iPEPS from .jld2 files conviently by:

    using JLD2
    save("/YourPath/Save.jld2", "ipeps", ipeps, "envs", envs)
    ipeps, envs = load("/YourPaht/Save.jld2", "ipeps", "envs")

We also provide `check_qn(ipeps::iPEPS, envs::iPEPSenv)` and `check_qn(ipeps::iPEPSΓΛ)` methods for debug. They would throw warnings when the ipeps or environment tensors 
have mismatched quantum spaces, which can be useful when dealing with symmetric iPEPS tensors.

    
