using HypothesisTests
using PlotlyJS
using Distributed
addprocs()
@everywhere begin
    using CSV
    using DataFrames
    using Distributions
    using Combinatorics
    using DataStructures
    using DataFrames
    using StatsBase
    using RCall
end

@everywhere R"""
suppressMessages(suppressWarnings(library(ROCR)))
suppressMessages(suppressWarnings(library(BayesFactor)))
"""

@everywhere begin
    function sho_var(v, X...)
        ind=∩(X...)
        s1 =sum(v[ind])
        s2 = *([ sum(v[x]) for x in X]...)
        return (s1 / s2)-1
    end

    #Olsson-Glass measure of coherence
    ols_var(v::Vector{Float64},X...) = (sum(v[∩(X...)])/sum(v[union(X...)]) - 0.5)
    # we do not care about actual values here, rather about the discrimination of true and false sets, hence we try to push values of coherence to negative for incoherent sets and positive for coherent sets
    # this does not generally hold for the standard Olsson-Glass measure, but it does not affect the results anyway. The same is used for below measures (with +0.5 to get to the unadjusted values where necessary when ols_var is used as a component).

    # Our generalisation of Olsson-Glass measure
    function ols2_var(v::Vector{Float64},X...)
        numerator=1
        denumerator_delta=1
        for i in X
            numerator*=sum(v[union(i)])
            denumerator_delta*=sum(v[setdiff(collect(1:length(v)),i)])
        end
        ols_ind=(numerator/(1-denumerator_delta))
        return (((ols_var(v,X...)+0.5)/ols_ind)-1)
    end

    # Meijs's (2006) generalisation of Olsson-Glass measure
    function olssharp_var(v::Vector{Float64},X...)
        ss=collect(powerset(X,2))
        as=[]
        for subs in ss
            a=ols_var(v,subs...)+0.5
            push!(as,a)
        end
        return mean(as) -0.5
    end

    # splitting up the set in appropriate parts (for the next function)
    function nonoverlappingsubsets_v2(X)
        X2=[]
        for nr in 1:length(X)
            push!(X2,nr)
        end
        ss = collect(powerset(X2,1))
        nonoverlaps = []
        ss2=[]

        for subset_1 in ss
            for subset_2 in ss
                if isempty(intersect(subset_1,subset_2))
                    if ~([subset_1,subset_2] in nonoverlaps) && ~([subset_2,subset_1] in nonoverlaps)
                        push!(nonoverlaps,[subset_1,subset_2])
                    end
                end
            end
        end
        nonoverlaps2=[]
        for pair in nonoverlaps
            delta1=[]
            delta2=[]
            for i in pair[1]
                push!(delta1,X[i])
            end
            for j in pair[2]
                push!(delta2,X[j])
            end
            push!(nonoverlaps2,[delta1,delta2])
        end

        return nonoverlaps2
    end

    #generalised Olsson-Glass measure (Koscholke, Schippers, Stegmann, 2019)
    function olsstar_var(v::Vector{Float64},X...)
        pairs=nonoverlappingsubsets_v2(X)
        as=[]
        for pair in pairs
            push!(as,ols_var(v,intersect(pair[1]...),intersect(pair[2]...)) + 0.5)
        end
        return mean(as) -0.5
    end

end

@everywhere function reuse_sim_data(n_prop::Int64=3,a0::Float64=0.5,n_worlds::Int64=5,nrsim::Int64=1)
    filename = string("./simdata-n",n_prop,"-a0_",a0, "_n_worlds",n_worlds,"simnr_",nrsim,".csv")
    csv_reader= CSV.File(filename)
    res2=[]
    for run in csv_reader
# calculate overlap measures (ol1: standard Olsson-Glass, ol2: our generalization, ol3: Meijs' 2005 generalisation, ol4: Koscholke et al 2019 generalization) and return how many propositions were true
        trueprops=Int64(run[1])
        prob_distr = parse.(Float64, split(run[2][2:end-1], ", "))
        which_Pworlds = eval.(Meta.parse.(replace(replace(run[3], "Any[" => "["), "]" => "]")))
        ol1 = ols_var(prob_distr,which_Pworlds...)
        ol2 = ols2_var(prob_distr,which_Pworlds...)
        ol3 = olssharp_var(prob_distr,which_Pworlds...)
        ol4 = olsstar_var(prob_distr,which_Pworlds...)
        shog = sho_var(prob_distr,which_Pworlds...)
        push!(res2,tuple(trueprops,ol1,ol2,ol3,ol4,shog))
    end

    df_cf = DataFrame(res2)
    dfs=[]
    trueEnoughProps=[n_prop-2,n_prop-1,n_prop]
    if n_prop <= 4
        trueEnoughProps=[n_prop-1,n_prop]
    end
    for trueEnough in trueEnoughProps
        pos=1
        a=df_cf[:,:]
        for trueProp in a[!,1]
            if trueProp>=trueEnough
                a[pos,1]=true
            else
                a[pos,1]=false
            end
            pos+=1
        end
        a[!,:1] = convert.(Bool,a[!,:1])
        logreg=[ lrMod(i, a)[1] for i in 1:size(a, 2) - 1 ]
        push!(dfs,logreg)
    end
    return dfs
end


function rerun_sim(numb_sim::Int64,a0::Float64,n_prop::Int64)
    trueEnoughProps=Array{Int64,1}([n_prop-2,n_prop-1,n_prop])
    if n_prop <= 4
        trueEnoughProps=[n_prop-1,n_prop]
    end
    outars1=Array{Any,3}(undef,length(trueEnoughProps),20,numb_sim)
    outars=[]
    for i in 1:numb_sim
        println(i)
        outars1[:,:,i] = @distributed (hcat) for n in 5:5:100
            reuse_sim_data(n,a0,n_prop,i)
        end
    end
    for thresh in 1:length(trueEnoughProps)
        temp=Array{Float64,3}(undef,4,20,numb_sim)
        for i in 1:numb_sim
            temp[:,:,i]=hcat(outars1[thresh,:,i]...)
        end
        push!(outars,temp)
    end
    return outars,trueEnoughProps # outars[1] is out_ar for first trueEnoughPerc etc
end

function aucPlot(name::String,out_auc::Array{Float64,3},trueenoughprops::Int64,n_prop::Int64,a0::Float64)
    range1=n_prop*2
    while range1%5>0
        range1+=1
    end
    titula="True enough pieces of info: "*string(trueenoughprops)*" (of "*string(n_prop)*"), prior="*string(a0)
    trace1  = scatter(x=5:5:100, y=out_auc[1, :], mode="markers+lines", name="OG",marker=attr(symbol=0),line=attr(dash="dot"))
    trace2  = scatter(x=5:5:100, y=out_auc[2, :], mode="markers+lines", name="OG<sup>+</sup>",marker=attr(symbol=1))
    trace3  = scatter(x=5:5:100, y=out_auc[3, :], mode="markers+lines", name="OG'",marker=attr(symbol=2),line=attr(dash="dashdot"))
    trace4  = scatter(x=5:5:100, y=out_auc[4, :], mode="markers+lines", name="OG<sup>*</sup>",marker=attr(symbol=3),line=attr(dash="dash"))
    trace4  = scatter(x=5:5:100, y=out_auc[5, :], mode="markers+lines", name="Shog",marker=attr(symbol=4),line=attr(dash="dot"))
    layout  = Layout(width=850, height=510, margin=attr(l=80, r=10, t=50, b=70),
                     xaxis=attr(title="Number of worlds", tickfont=attr(size=18),range=[range1,100]), yaxis=attr(tickfont=attr(size=18)), font_size=20,
                     annotations=[(x=-0.15, y=.5, xref="paper", yref="paper", text="AUC", showarrow=false, textangle=-90, font=Dict(size=>21))],title=titula)
    data    = [trace1,trace3,trace4,trace2]
    a=Plot(data, layout)
    if name[1] == '.' # if we are saving plots into current working directory, obtainable by > pwd(), changeable by cd("path")
        savefig(a,name)
    end
end

function arr_to_csv(x, outputstring)
    df = DataFrame(i = Float64[], j = Float64[], k = Float64[], x = Float64[])
    sizes = size(x)

    for k in 1:sizes[3]
        for j in 1:sizes[2]
            for i in 1:sizes[1]
                push!(df, (i, j, k, x[i,j,k]))
            end
        end
    end
    df |> CSV.write(outputstring, header = ["measure", "worlds", "repetition", "value"])
end

function rerun_sims(numb_sim::Int64,a0s::Array{Float64,1},n_prop::Int64)

    for a0 in a0s
        println("a0")
        println(a0)
        results=rerun_sim(numb_sim,a0,n_prop)
        out_ars = results[1]
        trueenprop = results[2]
        out_aucs=[]
        for prop in 1:length(trueenprop)
            out_auc = mean(out_ars[prop], dims=3);
            push!(out_aucs,out_auc)
        end

        for (i,j) in zip(trueenprop,1:length(trueenprop))
            name = string("./auc-n",n_prop,"-a0_", a0, "confirmed_prop_autoregenerated_",string(i),"_trueenough_prop.pdf")
            aucPlot(name,out_aucs[j],i,n_prop,a0)
            arr_to_csv(out_ars[j],string("./auc-n",n_prop,"-a0_",a0, "confirmed_prop_autoregenerated_",string(i),"_trueenough_propfull.csv"))
            arr_to_csv(out_aucs[j],string("./auc-n",n_prop,"-a0_",a0, "confirmed_prop_autoregenerated_",string(i),"_trueenough_prop.csv"))
        end
    end
end
