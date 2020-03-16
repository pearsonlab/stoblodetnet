# data := a vector of T adjacency matrices: A = [a1, ..., aT]
# z := integer vector where z[i] is the node type of node i, z[i] ∈ 1,2,...,K
# ρ := float64 vector where edge of type (a,b), a<=b, has strength of repulsion ρ[(b choose 2)+a]
# q* := float64 vector where edge of type (a,b), a<=b, has quality* q*[(b choose 2)+a]
### priors:
## z[i] ~ ceiling(uniform(0,K))
## q[i] ~ Gamma(α=.0001, β=.0001)
## ρ[i] ~ Uniform(0,1/2)
###

using Distributions, LinearAlgebra, Combinatorics, SparseArrays

## data
n = 10;
K_true = 3;
T = 20;
ρ_true = [.1,.2,.25,.3,.4,.45];
# ρ_true = rand_ρ(n);
q_true = [.1,.2,.3,.4,.5,.6];
# q_true = rand_q(n);
z_true = rand_z(n,K_true);
A = simulate_samp(n,T,z_true,ρ_true,q_true,K_true);
##
## resulting priors
n = size(A[1])[1];
# K = size(A[1],1);
K = K_true;
z = [ceil(Int,x) for x in K*rand(n)];
n_edge = binomial(n,2);
ρ = rand(binomial(K+1,2))*.5;
q = rand(Gamma(0.5,0.0001),n_edge);
q = rand(Gamma(0.5,0.0001),binomial(K+1,2));
q = q./maximum(q).-.000000001;
# q = rgamma(n_edge, shape = .0001, scale = 1/.0001);
##
###

# test update_z
S = 5000;
for s in 1:S
    global z = update_z(z,A,ρ_true,q_true,K);
end
#

# test update_ρq
S = 5000;
# i = 0;
for s in 1:S
    global ρ, q = update_ρq(z_true,A,ρ,q,K);
#     global i = i+1;
end
#

### sampler
S = 500;
i = 0;
for s in 1:S
    global z = update_z(z,A,ρ,q,K);
    global ρ, q = update_ρq(z,A,ρ,q,K);
    global i = i+1;
end
###


### data simulater
## simulater
# n := number of nodes
# T := number of adjacency matrices
# ρ := true vector where edge of type (a,b), a<=b, has strength of repulsion ρ[(b choose 2)+a]
# q := true vector where edge of type (a,b), a<=b, has quality* q*[(b choose 2)+a]
function simulate_samp(n::Int,T::Int,z::Vector{Int},ρ::Vector{Float64},q::Vector{Float64},K::Int)
    S = StoBloDetNet(n,z,K,ρ,q);
    samp = rand(S,T);
    A = [a_edge(s,n) for s in samp];
end

# Transform a set of edge into an adjacency matrix
function a_edge(edge::Array{Tuple{Int64,Int64},1},n::Int)
    a = rand(0:0,n,n);
    for e in edge
        a[CartesianIndex(e)] = 1;
    end
    a = Symmetric(a);
    return a
end

function rand_q(K::Int)
    res = rand(binomial(K+1,2));
    res[res.==1] .= 0;
    return res
end

function rand_ρ(K::Int)
    res = .5*rand(binomial(K+1,2));
    return res
end

function rand_z(n::Int,K::Int)
    res = rand(1:K,n);
    return res
end

#### help functions
## slice sampler for ρ and q
## q[i] ~ Gamma(α=.5, β=.0001)
## ρ[i] ~ Uniform(0,1/2)
# the function below does not deal with the boundedness issue
# function update_ρq(z::Vector{Int},A::Vector{Symmetric{Int,Matrix{Int}}},ρ::Vector{Float64},q::Vector{Float64},K::Int)
#     w_ρ = .1;
#     w_q = .5; # initial width     ???good choices???
#     α = .5;
#     β = .0001;
#     for i in 1:length(ρ)
#         y = rand()*exp(postA(z,A,ρ,q,K))*(q[i]^(α-1)*exp(-β*q[i]));
#         # randomly position hyperrectangle
#         # for ρ 
#         ρ_i_old = ρ[i];
#         u_ρ = rand();
#         L_ρ = ρ_i_old - w_ρ*u_ρ;
#         R_ρ = L_ρ + w_ρ;
#         # for q
#         q_i_old = q[i];
#         u_q = rand();
#         L_q = q_i_old - w_ρ*u_q;
#         R_q = L_q + w_q;
#         # sample from H, shrinking when points are rejected
#         while true
#             u_ρ = rand();
#             u_q = rand();
#             ρ[i] = L_ρ + u_ρ*(R_ρ-L_ρ);
#             q[i] = L_q + u_q*(R_q-L_q);
#             P = exp(postA(z,A,ρ,q,K))*(q[i]^(α-1)*exp(-β*q[i]));
#             if (y<P)
#                 break
#             end
#             if ρ[i]<ρ_i_old
#                 L_ρ = ρ[i];
#             else
#                 R_ρ = ρ[i];
#             end
#             if q[i]<q_i_old
#                 L_q = q[i];
#             else
#                 R_q = q[i];
#             end
#         end
#     end
#     return ρ, q
# end
function update_ρq(z::Vector{Int},A::Vector{Symmetric{Int,Matrix{Int}}},ρ::Vector{Float64},q::Vector{Float64},K::Int)
    α = .5;
    β = .0001;
    for i in 1:length(ρ)
        y = rand()*exp(postA(z,A,ρ,q,K))*(q[i]^(α-1)*exp(-β*q[i]));
        # for ρ
        ρ_i_old = ρ[i];
        u_ρ = rand();
        L_ρ = .0;
        R_ρ = .5;
        # for q
        q_i_old = q[i];
        u_q = rand();
        L_q = .0;
        R_q = 1.0;
        # sample from H, shrinking when points are rejected
        while true
            u_ρ = rand();
            u_q = rand();
            ρ[i] = L_ρ + u_ρ*(R_ρ-L_ρ);
            q[i] = L_q + u_q*(R_q-L_q);
            P = exp(postA(z,A,ρ,q,K))*(q[i]^(α-1)*exp(-β*q[i]));
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
    return ρ,q
end

## Gibbs sampler to update node types z
## note: can be sped up dramatically by storing the loglik of each block and
## updating only for the changed blocks on each iteration of the sampler
function update_z!(z::Vector{Int},y::Vector{Tuple{Int,Int}},ρ::Vector{Float64},q::Vector{Float64},K::Int)

    logp_zi = Array{Float64,1}(undef,K);
    for i in 1:length(z)
        for k = 1:K
            z[i] = k;
            Szik = StoBloDetNet(length(z),z,K,ρ,q); #recreate complete L; inefficient (see above)
            logp_zi[k] = logpdf(Szik,y); #flat prior in z so only likelihood matters for posterior
        end
        logp_zi = logp_zi .- logsumexp(logp_zi); #normalize conditional posterior
        z[i] = rand(Categorical(exp.(logp_zi)); #sample new k
    end
end

# ## Metropolis sampler to update node types z
# function update_z(z::Vector{Int},A::Vector{Symmetric{Int,Matrix{Int}}},ρ::Vector{Float64},q::Vector{Float64},K::Int)
#     # METROPOLIS
#     for i in 1:length(z)
#         # prob before change
#         zi_old = z[i];
#         p_z_old = postz(z,A,ρ,q,K);
#         # sample new value for z[i] from flat prior
#         zi_new = ceil(K*rand());
#         # calculate numerator
#         z[i] = zi_new;
#         p_z_new = postz(z,A,ρ,q,K);
#         ##
#         r = p_z_new/p_z_old;
#         u = rand();
#         if u<r
#             z[i]=zi_new;
#         else
#             z[i]=zi_old;
#         end
#     end
#     return z
# end
## I don't think you need any of this? All you need is the likelihood
# function postz(z::Vector{Int},A::Vector{Symmetric{Int,Matrix{Int}}},ρ::Vector{Float64},q::Vector{Float64},K::Int)
#     # calculate numerator
#     num = exp(postA(z,A,ρ,q,K))*(1/K);
#     # calculate denomenator
#     den = 0;
#     for k in 1:K
#         z[k] = k;
#         # den += exp(sum([posta(z,a,ρ,q,K) for a in A]))*(1/K);
#         den += exp(postA(z,A,ρ,q,K))/K;
#     end
#     return num/den
# end

# function posta(z::Vector{Int},a::Symmetric{Int,Matrix{Int}},ρ::Vector{Float64},q::Vector{Float64},K::Int)
#     n = size(a,1);
#     S = StoBloDetNet(n,z,K,ρ,q);
#     edge = edge_a(a);
#     pa = logpdf(S,edge);
# end

# function posta(z::Vector{Int},a::Symmetric{Int,Matrix{Int}},ρ::Vector{Float64},q::Vector{Float64},K::Int)
#     # compute log normconst
#     n = size(a,1);  # number of nodes
#     L = gen_L(z,n,ρ,q,K); # WHY is NaN generated for L(i,j)??????
#     replace!(L,NaN=>0); # a temporary remedy to the above problem
#     normconst = logdet(L+I);
#     # get La
#     EDGE = collect(combinations(1:n,2));
#     EDGE = map(x -> tuple(x...),EDGE);
#     edge = edge_a(a);
#     ind = La_ind(EDGE,edge);
#     La = reshape(L[ind,ind],(length(edge),length(edge)));
#     # compute posta
#     pa = logdet(La)-normconst;
#     return pa
# end

function postA(z::Vector{Int},A::Vector{Symmetric{Int,Matrix{Int}}},ρ::Vector{Float64},q::Vector{Float64},K::Int)
    res = 0;
    for i in 1:length(A)
        res+=posta(z,A[i],ρ,q,K);
    end
    return res
end

function La_ind(EDGE::Array{Tuple{Int64,Int64},1},edge::Array{Tuple{Int64,Int64},1})
    m = length(edge);
    ind = [findall(x->x==e,EDGE) for e in edge];
    ind = hcat(ind...);
    return ind
end

# # the gen_L function is recreated since edge types are taken into consideration
# function gen_L(z::Vector{Int},n::Int,ρ::Vector{Float64},q::Vector{Float64},K::Int)
#     edge = collect(combinations(1:n, 2));
#     N = length(edge);
#     L = Array{Float64,2}(undef,N,N);
#     d = q./(1 .-q);
#     for (i, ei) in enumerate(edge)
#         for j in i:N
#             if i == j
#                 ind = edgetype(ei,z,K);
#                 L[j, i] = d[ind];
#             elseif ei[1] ∈ edge[j]
#                 if ei[1]==edge[j][1] && z[ei[2]]==z[edge[j][2]]
#                     #ind = et_ind(ei);
#                     ind = edgetype(ei,z,K);
#                     L[j ,i] = ρ[ind]*d[ind];
#                 elseif ei[1]==edge[j][2] && z[ei[2]]==z[edge[j][1]]
#                     #ind = et_ind(ei);
#                     ind = edgetype(ei,z,K);
#                     L[j ,i] = ρ[ind]*d[ind];
#                 end
#             else
#                 L[j, i] = 0.0;
#             end
#         end
#     end
#     return L
# end

# function et_ind(e::Tuple{Int,Int})
#     z1 = z[e[1]]
#     z2 = z[e[2]]
#     if z1 < z2
#         ind = binomial(z2,2)+z1
#     else
#         ind = binomial(z1,2)+z2
#     end
#     return ind
# end

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

# function subMatrix(M::Array{Float64,2},row::Int,col::Int)
#     sub = M[setdiff(1:end,row),setdiff(1:end,col)];
#     return sub
# end
#
# function determinant(M::Array{Float64,2},n::Int)
#     det = 0;
#     if n==1
#         return M[1,1];
#     elseif n==2
#         return M[1,1]*M[2,2]-M[2,1]*M[1,2]
#     else
#         for i in 1:n
#             det = det+((-1)^(i+1) * M[1,i] * determinant(subMatrix(M,1,i),n-1));
#         end
#     end
#     return det
# end


####
