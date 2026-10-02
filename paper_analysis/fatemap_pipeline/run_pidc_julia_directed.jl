# Directed PIDC on top of NetworkInference.jl (0.1.0).
# The package's PUC score of a triple (x, y | target z) gives x->z and y->z a "proportional unique contribution"; it then adds every contribution to BOTH
# puc[x,z] and puc[z,x] (increment_puc_scores) and finally weights an unordered pair by cdf_i(score) + cdf_j(score) -> symmetric.
# Here: (1) the contribution is added only to puc[source, target] (row = source gene), i.e. the un-mirrored quantity, and
#       (2) the weight of source i -> target j is the gamma-CDF of puc[i,j] under the gamma fit to source i's own scores (row i, positive values),
#           with a normal fallback -- the same normalization the Python PIDC of run_competitors_fatemap.py uses.
# Check: puc + puc' must equal the package's own symmetric PUC matrix (printed as max abs difference).
using NetworkInference
using InformationMeasures
using Statistics
const Dist = NetworkInference.Distributions

path, outpath = string(ARGS[1]), string(ARGS[2])
nodes = get_nodes(path)
n = length(nodes); base = 2; estimator = "maximum_likelihood"
println("nodes: ", n, "  threads: ", Threads.nthreads())

function directed_puc(nodes, n, estimator, base)
    node_pairs = Array{NetworkInference.NodePair}(undef, n, n)
    for i in 1:n, j in i+1:n
        probabilities, p1, p2 = NetworkInference.get_joint_probabilities(nodes[i], nodes[j], estimator)
        mi = apply_mutual_information_formula(probabilities, p1, p2, base)
        si1 = apply_specific_information_formula(probabilities, p1, p2, 1, base)
        si2 = apply_specific_information_formula(probabilities, p2, p1, 2, base)
        node_pairs[i, j] = NetworkInference.NodePair(mi, si1)
        node_pairs[j, i] = NetworkInference.NodePair(mi, si2)
    end
    puc = zeros(n, n)
    Threads.@threads for z in 1:n            # each thread owns column z (target z), so no write conflicts
        for x in 1:n
            x == z && continue
            for y in x+1:n
                y == z && continue
                red = apply_redundancy_formula(nodes[z].probabilities, node_pairs[x, z].si, node_pairs[y, z].si, base)
                for (s, mi) in ((x, node_pairs[x, z].mi), (y, node_pairs[y, z].mi))
                    v = (mi - red) / mi
                    puc[s, z] += isfinite(v) ? v : 0.0
                end
            end
        end
    end
    return puc
end

@time puc = directed_puc(nodes, n, estimator, base)
if n <= 200
    ref = NetworkInference.get_puc_scores(nodes, n, estimator, base)
    println("check: max |(puc + puc') - package PUC| = ", maximum(abs.((puc .+ puc') .- ref)))
end

weights = zeros(n, n)
for i in 1:n
    idx = [j for j in 1:n if j != i]
    vals = puc[i, idx]
    F = try
        Dist.fit(Dist.Gamma, filter(v -> v > 0, vals))
    catch
        Dist.Normal(mean(vals), std(vals) > 0 ? std(vals) : 1.0)
    end
    for j in idx
        weights[i, j] = Dist.cdf(F, puc[i, j])
    end
end

open(outpath, "w") do io
    for i in 1:n, j in 1:n
        i == j && continue
        println(io, nodes[i].label, "\t", nodes[j].label, "\t", weights[i, j])
    end
end
println("wrote ", outpath)
