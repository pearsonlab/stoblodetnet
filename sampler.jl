# data := a vector of T adjacency matrices: A = [a1, ..., aT]
# z := integer vector where z[i] is the node type of node i, z[i] ∈ 1,2,...,K
# ρ := float64 vector where edge of type (a,b), a<=b, has strength of repulsion ρ[(b choose 2)+a]
# q* := float64 vector where edge of type (a,b), a<=b, has quality* q*[(b choose 2)+a]
### priors:
## z[i] ~ ceiling(uniform(0,K))
## q[i] ~ Gamma(α=.0001, β=.0001)
## ρ[i] ~ Uniform(0,1/2)
###

using Distributions, LinearAlgebra, Combinatorics


### data simulater
## simulater
# n := number of nodes
# T := number of adjacency matrices
# ρ := true vector where edge of type (a,b), a<=b, has strength of repulsion ρ[(b choose 2)+a]
# q := true vector where edge of type (a,b), a<=b, has quality* q*[(b choose 2)+a]
function simulate_samp(n::Int,T::Int,z::Vector{Int},ρ::Vector{Float64},q::Vector{Float64})
    A = Vector{Symmetric{Int,Matrix{Int}}}(undef,T);
    for i in 1:T
        a = ;
        append!(A[i],a);
    end
end
##
## data
ρ_true = ;
q_true = ;
z_true = ;
A = simulate_samp(10,20,z_true,ρ_true,q_true);
##
## resulting priors
n = size(A[1])[1];
z = [ceil(x) for x in n*rand(n)];
n_edge = binomial(n,2);
ρ = rand(n_edge)*.5;
q = rgamma(n_edge, shape = .0001, scale = 1/.0001);
##
###

### sampler
S = 2000;
for s in 1:S
    z = update_z(z,A,ρ,q);
    ρ, q = update_ρq(z,A,ρ,q);
end
###


#### help functions
## slice sampler for ρ and q
## q[i] ~ Gamma(α=.0001, β=.0001)
## ρ[i] ~ Uniform(0,1/2)
function update_ρq(z::Vector{Int},A::Vector{Symmetric{Int,Matrix{Int}}},ρ::Vector{Float64},q::Vector{Float64})
    w_ρ = 0.5/1000;
    w_q = 1; # initial width     ???good choices???
    α = .0001;
    β = .0001;
    for i in 1:length(ρ)
        y = rand()*exp(sum([posta(z,a,ρ,q) for a in A]))*(q[i]^(α-1)*exp(β*q[i]));
        # randomly position hyperrectangle
        # for ρ 
        ρ_i_old = ρ[i];
        u_ρ = rand();
        L_ρ = ρ_i_old - w_ρ*u_ρ;
        R_ρ = L_ρ + w_ρ;
        # for q
        q_i_old = q[i];
        u_q = rand();
        L_q = q_i_old - w_ρ*u_q;
        R_q = L_q + w_q;
        # sample from H, shrinking when points are rejected
        while true
            u_ρ = rand();
            u_q = rand();
            ρ[i] = L_ρ + u_ρ*(R_ρ-L_ρ);
            q[i] = L_q + u_q*(R_q-L_q);
            P = exp(sum([posta(z,a,ρ,q) for a in A]))*(q[i]^(α-1)*exp(β*q[i]));
            if (y<P)
                break
            end
            if ρ[i]<ρ_i_old
                L_ρ = ρ[i];
            else
                R_ρ = ρ[i];
            end
            if q[i]<q_i_old
                L_q = q[i];
            else
                R_q = q[i];
            end
        end
    end
    return ρ, q
end

## Metropolis sampler to update node types z
function update_z(z::Vector{Int},A::Vector{Symmetric{Int,Matrix{Int}}},ρ::Vector{Float64},q::Vector{Float64})
    # METROPOLIS
    K = size(A[1],1);
    for i in 1:length(z_prev)
        # prob before change
        zi_old = z[i];
        p_z_old = postz(z,A,ρ,q);
        # sample new value for z[i] from flat prior
        zi_new = ceil(K*rand());
        # calculate numerator
        z[i] = zi_new;
        p_z_new = postz(z,A,ρ,q);
        ##
        r = p_z_new/p_z_old;
        u = rand();
        if u<r
            z[i]=zi_new;
        else
            z[i]=zi_old;
        end
    end
    return z
end

function postz(z::Vector{Int},A::Vector{Symmetric{Int,Matrix{Int}}},ρ::Vector{Float64},q::Vector{Float64})
    # calculate numerator
    num = exp(sum([posta(z,a,ρ,q) for a in A]))*(1/K);
    # calculate denomenator
    den = 0;
    for k in 1:K
        z[i] = k;
        den += exp(sum([posta(z,a,ρ,q) for a in A]))*(1/K);
    end
    return num/den
end

function posta(z::Vector{Int},a::Symmetric{Int,Matrix{Int}},ρ::Vector{Float64},q::Vector{Float64})
    # compute log normconst
    n = size(a,1);  # number of nodes
    L = gen_L(z,binomial(n,2),ρ,q);
    normconst = logdet(L+I);
    # get La
    EDGE = collect(combinations(1:n,2));
    EDGE = map(x -> tuple(x...),EDGE);
    edge = edge_a(a);
    ind = La_ind(EDGE,edge);
    La = reshape(L[ind,ind],(length(edge),length(edge)));
    # compute posta
    pa = logdet(La)-normconst;
    return pa
end

function La_ind(EDGE::Array{Tuple{Int64,Int64},1},edge::Array{Tuple{Int64,Int64},1})
    m = length(edge);
    ind = [findall(x->x==e,EDGE) for e in edge];
    ind = hcat(ind...);
    return ind
end

function gen_L(z::Vector{Int},n::Int,ρ::Vector{Float64},q::Vector{Float64})
    edge = collect(combinations(1:n, 2));
    N = length(edge);
    L = Array{Float64,2}(undef,N,N);
    d = q./(1 .-q);
    for (i, ei) in enumerate(edge)
        for j in i:N
            if i == j
                L[j, i] = d;
            elseif ei[1] ∈ edge[j]
                if ei[1]==edge[j][1] && z[ei[2]]==z[edge[j][2]]
                    ind = et_ind(ei);
                elseif ei[1]==edge[j][2] && z[ei[2]]==z[edge[j][1]]
                    ind = et_ind(ei);
                end
                L[j ,i] = ρ[ind]*d[ind];
            else
                L[j, i] = 0.0;
            end
        end
    end
    return L
end

function et_ind(e::Tuple{Int,Int})
    z1 = z[e[1]]
    z2 = z[e[2]]
    if z1 < z2
        ind = binomial(z2,2)+z1
    else
        ind = binomial(z1,2)+z2
    end
    return ind
end

function edge_a(a::Symmetric{Int,Matrix{Int}})
    # generate the edges from an adjacency matrix
    edge = findall(x->x==1,a);
    edge = [i2s(e) for e in edge];
    edge = unique(edge);
    return edge
end

function i2s(e::CartesianIndex{2})
    # transform CartesianIndex to tuple{int,int}
    n1 = e[1];
    n2 = e[2];
    if n1 < n2
        t = tuple(n1,n2);
    else
        t = tuple(n2,n1);
    end
    return t
end
####
