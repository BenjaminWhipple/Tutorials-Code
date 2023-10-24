using StatsBase

function DifferentialEvolution(f,bounds,population_size,k_max;p=0.7,w=0.5,arguments=[])
    lb = bounds[1, :]
    ub = bounds[2, :]

    dim = size(bounds, 2)

    population = [lb .+ rand(dim) .* (ub - lb) for _ in 1:population_size]

    population = [clamp.(individual,lb,ub) for individual in population]

    n = length(population[1])
    m = length(population)
    for iter in 1:k_max
        if iter%10 == 0 println("Iter: "*string(iter)) end
        
        Threads.@threads for k=1:population_size
            a,b,c = sample(population, Weights([j!=k for j in 1:m]),3, replace = false)
            z = a + w*(b-c)
            bin_draw = rand(n)
            x_new = [ifelse(bin_draw[i]<p,z[i],population[k][i]) for i in 1:n]
            x_new = clamp.(x_new,lb,ub)
            if f(x_new,arguments...) < f(population[k],arguments...)
                population[k][:]=x_new
            end
        end

        population = [clamp.(individual,lb,ub) for individual in population]
        values = []
        for i in 1:length(population)
            push!(values,f(population[i],arguments...))
        end
        best=argmin(values)
        println(f(population[best],arguments...))
    end
    values = []
    for i in 1:length(population)
        push!(values,f(population[i],arguments...))
    end
    best=argmin(values)

    return f(population[best],arguments...), population[best]
end