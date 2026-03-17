using Distributions, LinearAlgebra, Random
using Statistics

include("CEM.jl")

# Ребро представляющее выходящий из узла с номером idx поток в содержащую данную структуру узел
# data - данные о потоке
mutable struct Edge
    idx::Int
    data::Vector{Union{Int, Nothing}}
end

# Узел с перечнем входящих в него рёбер
struct Node
    edges::Vector{Edge}
end

# Одно измерение потока для данного ребра
struct DataSingle
    idx_source::Int   # узел источник
    idx_target::Int   # узел назначение
    t::Int            # временная отметка
end

# Перечень требуемых измерений потока для некоторого ребра
mutable struct Context
    i::Int
    flows::Vector{DataSingle}
end

function get_edge(node::Node, idx::Int)
    edges = node.edges

    for i in 1:length(edges)
        if edges[i].idx == idx
            return edges[i]
        end
    end
    
    throw(DomainError(idx, "throw: get_edge"))
end

function context_build(road_graph::Vector{Node}, idx_source::Int, idx_target::Int, t_start::Int, m::Int)
    node_cnt = length(road_graph)
    (m > 0 || idx_source <= node_cnt && idx_target <= node_cnt) || error("error: context_build")
    
    flows = Vector{DataSingle}()
    
    edges = road_graph[idx_source].edges
    
    for i in 1:length(edges)
        idx = edges[i].idx
        if idx != idx_target                  # Обратный поток не учитываем
            for t in t_start:t_start + m - 1
                push!(flows, DataSingle(idx, idx_source, t))
            end
        end
    end
    
    return Context(1, flows)
end

function root_context_build(road_graph::Vector{Node}, idx_source::Int, idx_target::Int, d::Int, m::Int)
    node_cnt = length(road_graph)
    (idx_source <= node_cnt && idx_target <= node_cnt) || error("error: root_context_build")
    
    flows = Vector{DataSingle}()
    
    node_target = road_graph[idx_target]
    edge_target = get_edge(node_target, idx_source)
    
    for t in 1:d
        push!(flows, DataSingle(idx_source, idx_target, t))
    end
    
    edges_source = road_graph[idx_source].edges
    
    for i in 1:length(edges_source)
        idx = edges_source[i].idx
        if idx != idx_target                  # Обратный поток не учитываем
            for t in 1:m
                push!(flows, DataSingle(idx, idx_source, t))
            end
        end
    end
    
    return Context(1, flows)
end

function construct_E(road_graph::Vector{Node}, data_constructor::Vector{DataSingle})
    n = length(data_constructor)
    E = Vector{Int}(undef, n)
    
    for i in 1:n
        ds = data_constructor[i]
        node = road_graph[ds.idx_target]
        edge = get_edge(node, ds.idx_source)
        
        E[i] = edge.data[ds.t]
    end
    
    return E  
end

function construct_unit_data!(road_graph::Vector{Node}, data_constructor::Vector{DataSingle},
                              edge::Edge, t_start::Int, X::Vector{Vector{Int}})
    n = length(data_constructor)
    unit_data = Vector{Int}(undef, n + 1)
    
    k = length(edge.data)
    if t_start > k || edge.data[t_start] == nothing
        return
    end
    unit_data[1] = edge.data[t_start]
    
    for i in 1:n
        ds = data_constructor[i]
        t = t_start + ds.t
        node = road_graph[ds.idx_target]
        edge = get_edge(node, ds.idx_source)
        
        data = edge.data
        if t > length(data) || data[t] == nothing
            return
        end
        
        unit_data[i + 1] = data[t]
    end
    
    push!(X, unit_data)   
end

function data_build(road_graph::Vector{Node}, idx_source::Int, idx_target::Int, N::Int, d::Int=4, m::Int=5)
    l = length(road_graph)
    (idx_source <= l && idx_target <= l) || error("error: data_build")
    
    data_constructor = Vector{DataSingle}()  # Вектор задающий последовательность сборки единицы данных
    E = Vector{Int}()                        # Известные данные для прогнозирования
    
    root_context = root_context_build(road_graph, idx_source, idx_target, d, m)
    stack = [root_context]
    
    # Создание правила для сборки единицы данных(векторов)
    while !isempty(stack)
        stack_len = length(stack)
        context = stack[stack_len]
        
        flow_idx = context.i
        flows_len = length(context.flows)
        if flow_idx > flows_len
            pop!(stack)
            continue
        end
        
        flow_curr = context.flows[flow_idx]
        idx_source = flow_curr.idx_source
        idx_target = flow_curr.idx_target
        t = flow_curr.t
        
        edge = get_edge(road_graph[idx_target], idx_source)
        data_len = length(edge.data)
        if t > N
            throw(DomainError(t, "throw: data_build: данных недостаточно"))
        elseif t > data_len || edge.data[t] == nothing
            push!(stack, context_build(road_graph, idx_source, idx_target, t + 1, m))
        else
            push!(data_constructor, DataSingle(idx_source, idx_target, t))
        end
        
        context.i += 1
    end
    
    # Удаление повторяющихся элементов
    len = length(data_constructor)
    i = 1
    while i <= len
        a = data_constructor[i]
        j = i + 1
        while j <= len
            if a == data_constructor[j]
                deleteat!(data_constructor, j)
                len = length(data_constructor)
            else
                j += 1
            end
        end
        i += 1
    end

    E = construct_E(road_graph, data_constructor)

    data = Vector{Vector{Int}}()
    edge = get_edge(road_graph[idx_target], idx_source)
    
    for t_start = 1:N
        construct_unit_data!(road_graph, data_constructor, edge, t_start, data)
    end
    
    return E, data
end

function forecast(gmm, E, X)
    n_samples = size(E, 1)
    E_dim = size(E, 2)
    E_dim < size(X, 2) || error("error: gmm_forecast")
    
    k = gmm.n
    
    ferecasting = zeros(n_samples, size(X, 2)-E_dim) # Array{Float32}(undef, n_samples, size(X, 2)-E_dim)
    
    for j in 1:n_samples
#        ferecasting[j, :] .= 0.0
        norm_sum = 0.0
        for l in 1:k
            μ_lE = gmm.μ[l][end-E_dim+1:end]
            Σ_lEE = gmm.Σ[l][end-E_dim+1:end, end-E_dim+1:end] 
            
            β_l = gmm.α[l] * pdf(MvNormal(μ_lE, Σ_lEE), E[j, :])
            norm_sum += β_l
            
            μ_lF = gmm.μ[l][1:end-E_dim]
            Σ_lFE = gmm.Σ[l][1:end-E_dim, end-E_dim+1:end]
            
            μ_lF_E = μ_lF - Σ_lFE * inv(Σ_lEE) * (μ_lE - E[j, :])
            ferecasting[j, :] += β_l * μ_lF_E
        end
        
        ferecasting[j, :] ./= norm_sum
    end
    
    return ferecasting
end


# Граф дорог из A Bayesian network approach to traffic flow forecasting.pdf Fig. 1. (a)
"
              1        2        3
              +--------+--------+
              |        |        |
     4      5 |      6 |      7 |
     +--------+--------+--------+
              |        |        |
            8 |      9 |     10 |
              +--------+--------+         
"
road_graph = Vector{Node}()

push!(road_graph, Node([Edge(2, []), Edge(5, [])]))
push!(road_graph, Node([Edge(1, []), Edge(3, []), Edge(6, [])]))
push!(road_graph, Node([Edge(2, []), Edge(7, [])]))
push!(road_graph, Node([Edge(5, [])]))
push!(road_graph, Node([Edge(1, []), Edge(4, []), Edge(6, []), Edge(8, [])]))
push!(road_graph, Node([Edge(2, []), Edge(5, []), Edge(7, []), Edge(9, [])]))
push!(road_graph, Node([Edge(3, []), Edge(6, []), Edge(10, [])]))
push!(road_graph, Node([Edge(5, []), Edge(9, [])]))
push!(road_graph, Node([Edge(6, []), Edge(8, []), Edge(10, [])]))
push!(road_graph, Node([Edge(7, []), Edge(9, [])]))

N = 1400
M = 100000

for i in 1:length(road_graph)
    for j in 1:length(road_graph[i].edges)
        road_graph[i].edges[j].data = rand(1:M, N)
    end
end 
   
#road_graph[6].edges[1].data[2] = nothing
#road_graph[2].edges[1].data[7] = nothing
#road_graph[3].edges[1].data[2] = nothing

#road_graph[1].edges[2].data[8] = nothing

#road_graph[2].edges[1].data[1] = nothing
#road_graph[2].edges[2].data[1] = nothing
#road_graph[1].edges[2].data[2] = nothing
#road_graph[3].edges[2].data[2] = nothing

# Случай с повторяющимися элементами
road_graph[2].edges[1].data[1] = nothing
road_graph[1].edges[2].data[2] = nothing
road_graph[5].edges[3].data[3] = nothing

# Выборка данных для потока из узла 2 в узел 6
# E - вектор для прогнозирования
# data и X_training - данные для настройки параметров(обучения) GMM
src = 2
dst = 6

println("Прогноз для потока из узла $src в узел $dst")

E, data = data_build(road_graph, src, dst, N)
#E, data = data_build(road_graph, 10, 9, N)

println("Вектор для прогнозирования: $E")

X_training = Array{Int}(undef, length(data), length(E) + 1)

#println(size(X, 1))  # 1392
#println(size(X, 2))  # 46
    
for i in 1:size(X_training, 1)
    for j in 1:size(X_training, 2)
        X_training[i, j] = data[i][j]
    end
end

# Настройка параметров GMM
gmm = cem(X_training, 8, maxIter=70)

# Прогноз для потока из узла src в узел dst
ferecasting = forecast(gmm, E', X_training)

println("Прогноз для потока из узла $src в узел $dst: $ferecasting")

