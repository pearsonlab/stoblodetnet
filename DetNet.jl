# module StoBloDetNet

using LinearAlgebra, Combinatorics, NamedArrays, Distributions, StatsFuns
import Base: show, rand, getindex, length
import LinearAlgebra.eigen
import Distributions: logpdf, loglikelihood, params

# defines a single type "block" with edge-wise repulsion
# ρ: strength of repulsion, ∈ (0,0.5)
# q: "quality", in terms of marginal edge probability w/ no repulsion, ∈ [0,1)
# edges: vector of all possible (undirected) edges between nodes, (i,j), i<j
# L: the L-ensemble of the DPP
struct DetNet
    ρ::Float64
    q::Float64
    edge::Vector{Tuple{Int,Int}}
    L::Symmetric{Float64,Matrix{Float64}}

    function DetNet(edge::Vector{Tuple{Int,Int}},ρ::Float64,ω::Float64,q::Float64)
        n = length(edge);
        L = Array{Float64,2}(undef,n,n);
        d = q/(1-q);
        # this is not optimized but it's much faster than my attempt to do
        # something clever with iterators
        for (i, ei) in enumerate(edge)
            for j in i:n
                if i == j
                    L[j, i] = d;
                elseif ei[1] ∈ edge[j] || ei[2] ∈ edge[j]
                    L[j, i] = ρ*d;
                else
                    L[j, i] = ω*d;
                end
            end
        end
        return new(ρ,q,edge,Symmetric(L,:L))
    end
end

function DetNet(n::Int, ρ::Float64, ω::Float64, q::Float64)
    edge = collect(combinations(1:n, 2));
    edge = map(x -> tuple(x...),edge);
    return DetNet(edge,ρ,ω,q)
end

# function DetNet(n1, n2, ρ::Float64, q::Float64)
#     edge = vec(collect(Iterators.product(n1,n2)));
#     return DetNet(edge,ρ,q)
# end

# convenience functions
global const ∅ = Vector{Tuple{Int64,Int64}}(undef,0); #shorthand to define an empty set
eigen(S::DetNet) = eigen(S.L);
show(io::IO, x::DetNet) = show(io, x.L);
show(io::IO, mime::MIME"text/plain", x::DetNet) =
    show(io, mime, NamedArray(x.L, (string.(x.edge), string.(x.edge)) ));

getindex(S::DetNet, I) = S.L[I,I];
getindex(S::DetNet, E::Vector{Tuple{Int,Int}}) = getindex(S, indexin(E,S.edge));

# random sampling, hopefully self-explanatory
function __rand(eigdec)
    λ = eigdec.values;
    V = eigdec.vectors;

    p = λ ./ (λ .+ 1.);
    p[p .< 0] .= 0.; # hack for numerical issues
    whichvec = rand.(Bernoulli.(p)) .== 1;
    V = V[:,whichvec];
    n = sum(whichvec);

    chosen = Vector{Int64}(undef,n);
    for i in 1:n
        q = sum(abs2.(V), dims=2)./(n-i+1) |> vec;
        chosen[i] = rand(Categorical(q));
        V = V*svd(V[chosen[i]:chosen[i],:],full=true).V[:,2:(n-i+1)];
    end
    return chosen
end

function rand(S::DetNet, n::Int64)
    eigdec = eigen(S);
    return [S.edge[__rand(eigdec)] for i in 1:n];
end

rand(S::DetNet) = S.edge[__rand(eigen(S))];

# log probability of x, where x is a vector of edges representing a sample
# from the DPP (may be empty, but must be the correct type! use ∅)
function loglikelihood(S::DetNet, X::Vector{T}) where {T <: AbstractVector}
    normconst = logdet(S.L+I);
    return sum(logpdf(S, x, normconst) for x in X);
end

function logpdf(S::DetNet, x::T, normconst = logdet(S.L+I)) where {T <: AbstractVector}
    return logdet(S[x]) - normconst;
end

function logpdf(S::DetNet, X::Vector{T}, normconst = logdet(S.L+I)) where {T <: AbstractVector}
    return [logpdf(S, x, normconst) for x in X];
end
