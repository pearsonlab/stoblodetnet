include(string(pwd(),"/DetNet.jl"));
include(string(pwd(),"/StoBloDetNet.jl"));
using Plots
function sametype(z::Vector{T},K) where T
        return mapreduce(k -> (z.==k)*(z.==k)', +, 1:K)
end
function tril_vec(M)
        return M[tril!(trues(size(M)),-1)]
end

ρ = 0.499999;
q = 0.9;
K = 3;
n = 18;
z = repeat(collect(1:K),inner=div(n,K));
# z = vcat(fill(1,div(n*2,3)),fill(2,div(n,3)))
S = StoBloDetNet(n,z,K,ρ,q);
y = rand(S)

##### inference with known K
# zsamp = rand(Categorical(fill(1/K,K)), n);
zsamp = sample(z,n,replace=false);
niter = 10001;
thin = 2;
saveiter = 1:thin:niter;
nsamp = length(saveiter);
niter = maximum(saveiter);

sout = Matrix{Int}(undef,n,nsamp);
sout[:,1] = zsamp;
for t in 2:niter
        update_z!(zsamp,y,ρ,q,K);
        if t ∈ saveiter
                print(t,"\r")
                j = findfirst(saveiter.==t);
                sout[:,j] = zsamp;
        end
end

###### inference on K
Kseq = 2:5;
niter = 1001;
thin = 1;
saveiter = 1:thin:niter;
nsamp = length(saveiter);
niter = maximum(saveiter);

ll = Matrix{Float64}(undef,length(Kseq),nsamp);
for (i,k) in enumerate(Kseq)
        print(string(k),"\n")
        zsamp = rand(Categorical(fill(1/k,k)), n);
        ll[i,1] = logpdf(StoBloDetNet(n,zsamp,k,ρ,q),y);
        for t in 2:niter
                update_z!(zsamp,y,ρ,q,k);
                if t ∈ saveiter
                        print(t,"\r")
                        j = findfirst(saveiter.==t);
                        ll[i,j] = logpdf(StoBloDetNet(n,zsamp,k,ρ,q),y);
                end
        end
        print("\n")
end

plot(hcat(fill(logpdf(DetNet(n,ρ,q),y),nsamp),ll',fill(logpdf(S,y),nsamp)),
        label=hcat("1",string.(Kseq)...,"truth"),legend=:bottomright,
        xlab="iteration",ylab="loglik")

#### evaluate inference
typeA = [cor(tril_vec(sametype(z,K)),tril_vec(sametype(sout[:,i],K)))
                for i in 1:nsamp];
ll = [logpdf(StoBloDetNet(n,sout[:,i],K,ρ,q),y) for i in 1:nsamp];
p1 = plot(
        plot(saveiter,hcat(ll,fill(logpdf(S,y),nsamp)),yaxis=("loglik"),legend=false,
                title=string("rho=",ρ,", q=",q,", n=",n,", K=",K)),
        plot(saveiter,typeA,yaxis=("type agreement"),xaxis=("iteration"),legend=false),
        layout=(2,1));
plot(p1,heatmap(mean(sametype(sout[:,i],K) for i in 500:nsamp),aspect_ratio=:equal,
                xticks=div(n,K):div(n,K):n,yticks=div(n,K):div(n,K):n))


#### discriminability
n = 30;
z = repeat(collect(1:K),inner=div(n,K));
K = 3;
ρseq = [0.4, 0.45, 0.49, 0.499, 0.5-1e-5];
qseq = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99];
lldiff = Matrix{Float64}(undef,length(ρseq),length(qseq));
for (i,ρ) in enumerate(ρseq)
        for (j,q) in enumerate(qseq)
                S = StoBloDetNet(n,z,K,ρ,q);
                Y = rand(S,25);
                lldiff[i,j] = mean((logpdf(S,y)-
                        logpdf(StoBloDetNet(n,sample(z,n,replace=false),K,ρ,q),y)
                        for y in Y))
        end
end
heatmap(string.(qseq),string.(ρseq),lldiff, aspect_ratio=1,xaxis=("q"),yaxis=("rho"))
