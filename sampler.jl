# data := a vector of T adjacency matrices: A = [a1, ..., aT]
# z := integer vector where z[i] is the node type of node i, z[i] ∈ 1,2,...,K
# ρ := float64 vector where edge of type (a,b), a<=b, has strength of repulsion ρ[(b choose 2)+a]
# q* := float64 vector where edge of type (a,b), a<=b, has quality* q*[(b choose 2)+a]


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
function update_z!(z::Vector{Int},y::Vector{Tuple{Int,Int}},ρ,q,K)

    logp_zi = Array{Float64,1}(undef,K);
    for i in 1:length(z)
        for k = 1:K
            z[i] = k;
            Szik = StoBloDetNet(length(z),z,K,ρ,q); #recreate complete L
            logp_zi[k] = logpdf(Szik,y); #flat prior in z so only likelihood matters for posterior
        end
        logp_zi = logp_zi .- logsumexp(logp_zi); #normalize conditional posterior
        z[i] = rand(Categorical(exp.(logp_zi))); #sample new k
    end
end

function La_ind(EDGE::Array{Tuple{Int64,Int64},1},edge::Array{Tuple{Int64,Int64},1})
    m = length(edge);
    ind = [findall(x->x==e,EDGE) for e in edge];
    ind = hcat(ind...);
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
