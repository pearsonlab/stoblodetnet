# defines a block-structured DPP network where edges only repell when they
# connect nodes of the same type.
# edge: set of all edges in the network
# z: Interger vector where z[i] is the node-type of node i, z[i] ∈ 1,..,K
# block: Vector of DetNets, one for each unique combination of node-types
# K: number of possible node-types
struct StoBloDetNet
    edge::Vector{Tuple{Int,Int}}
    z::Vector{Int}
    block::Vector{DetNet}
    K::Int

    function StoBloDetNet(edge, z, K, ρ, q)
        et = [edgetype(e, z, K) for e in edge];
        block = [DetNet(edge[et .== i], ρ[i], q[i])
                    for i in 1:binomial(K+1,2)];
        new(edge,z,block,K)
    end
end

function StoBloDetNet(n::Int,z,K,ρ,q)
    edge = collect(combinations(1:n, 2));
    edge = map(x -> tuple(x...),edge);
    return StoBloDetNet(edge,z,K,ρ,q);
end

StoBloDetNet(n::Int,z,K,ρ::Float64,q::Float64) =
    StoBloDetNet(n,z,K,fill(ρ,binomial(K+1,2)),fill(q,binomial(K+1,2)));

# get the edge-type of e from e's node-types
edgetype(e, z, K) = div(z[e[1]],K) + z[e[2]];
edgetype(e, S::StoBloDetNet) = edgetype(e, S.z, S.K);

function rand(S::StoBloDetNet, n::Int)
    samp = [Vector{Tuple{Int,Int}}(undef,0) for i in 1:n];
    tmp = [rand(B,n) for B in S.block];
    for i in 1:n
        for k in 1:binomial(S.K+1,2)
            append!(samp[i],tmp[k][i]);
        end
    end
    return samp
end

rand(S::StoBloDetNet) = vcat([rand(B) for B in S.block]...);

function loglikelihood(S::StoBloDetNet, X::Vector{T}) where {T <: AbstractVector}
    normconst = [logdet(B.L+I) for B in S.block];
    return sum(logpdf(S,x,normconst) for x in X);
end

function logpdf(S::StoBloDetNet, x::T,
    normconst = [logdet(B.L+I) for B in S.block]) where {T <: AbstractVector}

    #number of edge types
    Ke = binomial(S.K+1,2);
    #get edge type for each element in x
    et = [edgetype(e,S) for e in x];
    #iterator for pulling out only edges of a given edge-type
    xiter = (x[et .== k] for k in 1:Ke);
    #loop over edge-types, evaluate
    return sum(logpdf(B, xk, nc) for (B, xk, nc) in zip(S.block, xiter, normconst));
end
