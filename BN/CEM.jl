using Distributions, LinearAlgebra, Random
using Statistics

mutable struct GMM
    n::Int
    α::Vector{Float64}
    μ::Vector{Vector{Float64}}
    Σ::Vector{Matrix{Float64}}
    gaussians::Vector{MvNormal}
end

function gmm_build(num_components, data)
    n = num_components
    α = fill(1.0 / n, n)  # Одинаковые априорные вероятности.
    
#    mean_data = mean(data, dims=1) # Среднее по каждому признаку
#    μ = [vec(mean_data) for _ in 1:n]  # data[inxs, :]
    μ = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        μ[i] = rand(size(data, 2))
    end
    
    cov_common = cov(data, dims=1) # + 0.01 * I # Одинаковые ковариационные матрицы.
    Σ = [cov_common for _ in 1:n]
    
    gaussians = Vector{MvNormal}(undef, n)
    
    for i in 1:n
        gaussians[i] = MvNormal(μ[i], Σ[i])
    end
    
    return GMM(n, α, μ, Σ, gaussians)
end

function gmm_clone(gmm)
    return deepcopy(gmm)
end 

function gmm_delete_component!(gmm, idx)
    gmm.n >= idx || error("error: gmm_delete_component!")
    
    gmm.n = gmm.n - 1
    
    deleteat!(gmm.α, idx)
    deleteat!(gmm.μ, idx)
    deleteat!(gmm.Σ, idx)
    deleteat!(gmm.gaussians, idx)
end

function gmm_split!(gmm, idx, data, max_mod_diff)
    gmm.n >= idx || error("error: gmm_split!")

    gmm.n = gmm.n + 1

    w = gmm.α[idx] / 2
    gmm.α[idx] = w;
    insert!(gmm.α, idx + 1, w)
    
    insert!(gmm.μ, idx + 1, vec(mean(data, dims=1))) # Пока не random
    
    σ0 = 0.15 * max_mod_diff
    A = σ0 * σ0 * Matrix{eltype(gmm.Σ[idx])}(I, size(gmm.Σ[idx],1), size(gmm.Σ[idx],2))
    gmm.Σ[idx] = A
    insert!(gmm.Σ, idx + 1, A)
    
    gmm.gaussians[idx] = MvNormal(gmm.μ[idx], gmm.Σ[idx])
    insert!(gmm.gaussians, idx + 1, MvNormal(gmm.μ[idx + 1], gmm.Σ[idx + 1]))
end

function gmm_merge!(gmm, m, l)
    gmm.n >= m && gmm.n >= l || error("error: gmm_merge!")

    if m == l
        return
    end
    
    α_m, α_l = gmm.α[m],  gmm.α[l]
    μ_m, μ_l = gmm.μ[m],  gmm.μ[l]

    gmm.n = gmm.n - 1

    α_M = α_m + α_l
    gmm.α[m] = α_M
    deleteat!(gmm.α, l)
    
    μ_M = (α_m * μ_m + α_l * μ_l) / α_M
    gmm.μ[m] = μ_M
    deleteat!(gmm.μ, l)
    
    μ_m_diff, μ_l_diff = μ_m - μ_M, μ_l - μ_M
    
    Σ_m, Σ_l = gmm.Σ[m], gmm.Σ[l]
    Σ_M = (α_m * (Σ_m + (μ_m_diff * μ_m_diff')) + α_l * (Σ_l + (μ_l_diff * μ_l_diff')))
    gmm.Σ[m] = Σ_M
    deleteat!(gmm.Σ, l)
    
    gmm.gaussians[m] = MvNormal(μ_M, Σ_M)
    deleteat!(gmm.gaussians, l)
end

function kl_divergence(gaussian, P, data) 
    n = size(data, 1)
    n == size(P, 1) || error("error: kl_divergence")
    
    kl_div = 0.0
    for i in 1:n
        kl_div = kl_div - P[i] * log(pdf(gaussian, data[i, :]))
    end
        
    if kl_div < 0
#        println(kl_div)
        kl_div = 0 # error("error: negative kl divergence")
    end
    
    return kl_div
end

function L_Figueiredo(gmm, x, length_data)
    N = 2 * gmm.n
    n = length_data
    k = gmm.n
    p_sum, α_sum = 0.0, 0.0
    
    for i in 1:k
        p_sum = p_sum + gmm.α[i] * pdf(gmm.gaussians[i], x)
        α_sum = α_sum + log(n * gmm.α[i] / 12)
    end
    
    return log(p_sum) - N*α_sum/2 - (k/2)*log(n/12) - k*(N + 1)/2
end
    

function acceptance_probability(gmm_prev, gmm_new, data)
    γ = 10
    n = size(data, 1)
    
    i = 1
    min_value = minimum([L_Figueiredo(gmm_new, data[i, :], n) - L_Figueiredo(gmm_prev, data[i, :], n) for i in 1:n])
    
    accept_level = exp(min_value / γ)
    u = rand(Uniform(0, 1))
    
    return u <= accept_level
end

function e_step(gmm, data)
    n_samples, n_features = size(data)
    k = gmm.n
    γ = zeros(n_samples, k)
    
    for i in 1:n_samples
        x = data[i, :]
        sum = 0.0
        ps = zeros(k)
        for j in 1:k
            p = gmm.α[j] * pdf(gmm.gaussians[j], x)
            ps[j] = p
            sum += p
        end
        γ[i, :] = ps ./ sum
    end
    
    return γ
end

function m_step!(gmm, data, γ)
    n_samples, n_features = size(data)
    k = gmm.n
    
    size(γ, 2) == k || error("error: m_step!")
    
    full_sum = sum(γ, dims=1)
    
    gmm.α = vec(full_sum) ./ n_samples
    
    for j in 1:k
        weighted_sum = zeros(n_features)
        for i in 1:n_samples
            weighted_sum += data[i, :] * γ[i, j]
        end
        gmm.μ[j] = weighted_sum / full_sum[j]
        
        Σ = zeros(n_features, n_features)
        for i in 1:n_samples
            diff = data[i, :] - gmm.μ[j]
            Σ += γ[i, j] * (diff * diff')
        end
        gmm.Σ[j] = Σ / full_sum[j] # + 0.1 * I
        
        gmm.gaussians[j] = MvNormal(gmm.μ[j], gmm.Σ[j])
    end
    
end

function loglikelihood(gmm, data)
    n_samples = size(data, 1)
    k = gmm.n
    
    log_L = 0.0
    
    for i in 1:n_samples
        p_j = 0.0
        for j in 1:k
            p_j += gmm.α[j] * pdf(gmm.gaussians[j], data[i, :])
        end
        log_L += log(p_j)
    end
    
    return log_L
end

struct split_cxt
    i::Int
    J_split::Float64
end

function J_split_calc(gmm, P, data) 
    k = gmm.n
    k > 0 || error("error: J_split_calc")
    
    J_max = split_cxt(0, -Inf)
    
    for m in 1:k
        kl_div = kl_divergence(gmm.gaussians[m], P, data)
        J_m = split_cxt(m, kl_div)
        
        if J_m.J_split > J_max.J_split
            J_max = J_m
        end
    end
    
    return J_max
end

struct merge_cxt
    i::Int
    j::Int
    J_merge::Float64
end

function J_merge_calc(gmm, γ, X)
    β = 3500000 #50000 # 10
    
    k = gmm.n
    k > 0 || error("error: J_merge_calc")
    
    J_max = merge_cxt(0, 0, -Inf)
    
    for i in 1:k
        for j in i+1 : k
            g = gmm_clone(gmm)
            gmm_merge!(g, i, j)
            kl_div = kl_divergence(g.gaussians[i], γ, X)
            J_ij = merge_cxt(i, j, β/kl_div)
            
            if J_ij.J_merge > J_max.J_merge
                J_max = J_ij
            end
        end
    end
    
    return J_max
end

function annihilation!(gmm, data, logL)
    n_samples = size(data, 1)
    
    while true
        try_annihilate = false
        k = gmm.n
        for m in 1:k
            if n_samples * gmm.α[m] < 2 * k
                gmm_delete_component!(gmm, m)
                m_step!(gmm, data, e_step(gmm, data))
                logL = loglikelihood(gmm, data)
                try_annihilate = true
                println("Аннигиляция")
                break
            end
        end

        if try_annihilate == false
            break
        end
    end
    
    return logL
end
        
function gmm_operation(gmm, γ, X, max_mod_diff)
    Js = J_split_calc(gmm, γ, X)
    println("kl_divergence для разделения: ", Js)
    
    Jm = J_merge_calc(gmm, γ, X)
    println("kl_divergence для слияния: ", Jm)
    
    gmm_new = gmm_clone(gmm)
    if Js.J_split >= Jm.J_merge
        println("Выбрана операция разделения")
        gmm_split!(gmm_new, Js.i, X, max_mod_diff)
    else
        println("Выбрана операция слияния")
        gmm_merge!(gmm_new, Jm.i, Jm.j)
    end
    
    if acceptance_probability(gmm, gmm_new, X)
        println("Операция принята")
        return gmm_new
    else
        println("Операция отклонена")
        return gmm
    end
end

function cem(X, k; maxIter=:1000, eps=:1e-4)
    i = 1
    max_mod_diff = maximum(
        [maximum(abs.(X[i, :] .- X[j, :])) for i in 1:size(X, 1), j in i+1:size(X, 1)]
    )

    println("Максимальный модуль разности: ", max_mod_diff)
    
    logL_old = -Inf

    gmm = gmm_build(k, X)

    for i in 1:maxIter
        γ = e_step(gmm, X)
        m_step!(gmm, X, γ)
        logL = loglikelihood(gmm, X)
        println("Итерация $i, Log Likelihood: $logL")
        
        if abs(logL - logL_old) < eps
            println("Максимум(возможно локальный) достигнут на итерации $i")
            gmm = gmm_operation(gmm, γ, X, max_mod_diff)
        end
        
        logL_old = annihilation!(gmm, X, logL)
    end

    return gmm
end

