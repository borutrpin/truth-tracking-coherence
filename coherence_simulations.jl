using HypothesisTests
using PlotlyJS
using Distributed
addprocs()
@everywhere begin
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
            if isempty(intersect(pair[1]...)) || isempty(intersect(pair[2]...))
                a=0
            else
                a=ols_var(v,intersect(pair[1]...),intersect(pair[2]...)) + 0.5
            push!(as,a)
            end
        end
        return mean(as) -0.5
    end

end

@everywhere function confSim_var_pre(ν::Int64,a0::Float64=0.3,n::Int64=3) # n = nr of pieces of information in a set
    @assert ν > 3 # no. of worlds with a minimum of 4 (in the actual analysis we look at minimum on 2n, where n is the number of pieces of information)
    which_Eworlds=[] # We want to know which worlds belong to pieces of evidence
    which_Pworlds=[] # Same for pieces of information.
    minimum_common_props=[] # We need at least one world in common to all pieces of information, otherwise the prior probability of the set would be necessarily 0.
    minimum_common_e=[] # Same for pieces of evidence: we assume that evidence is veristic and select the actual world from their intersection, so Es have to have at least one common possible worlds.
    probabilizedw=[] # We will need to keep in mind which worlds have been probabilized already.
    for information_piece in 1:n
        howmanyP=rand(1:ν-1) # how many possible worlds under a piece of information: from 1 to ν-1 to avoid contradictions and tautologies
        howmanyE=rand(1:ν-1) # same for pieces of evidence
        if minimum_common_props==[] # for the first piece of information which will have at least one world in common with the incoming
            whichP=sample(1:ν, howmanyP, replace=false) # which worlds will be in this piece of information
            push!(minimum_common_props,sample(whichP))
            push!(which_Pworlds,whichP)
            push!(minimum_common_e,sample(whichP)) #at the same time minimum common core for pieces of evidence and e1 and p1 (all e and p have to have at least one common world because e confirms p).
            e2=sample(setdiff(1:ν,minimum_common_e), howmanyE-1, replace=false)
            whichE=union(minimum_common_e,e2) # the first piece of evidence may also contain other possible worlds: we make sure not to duplicate.
            push!(which_Eworlds,whichE)
        else # for all other pieces of information
            p2=sample(setdiff(1:ν,minimum_common_props), howmanyP-1, replace=false) # we select worlds so that the common P-world is not included
            push!(which_Pworlds,union(minimum_common_props,p2)) # we then add the union of p2 and the common P-world to our array of worlds in various P (note: p2 might be empty if howmanyP = 1)
            e1=sample(last(which_Pworlds)) # we select one of the P-worlds from P generated in this loop and select a world that it will have in common with the corresponding piece of evidence (for confirmation)
            if e1==minimum_common_e
                minus=1
            else
                minus=2
            end
            if howmanyE!=1
                e2=sample(setdiff(1:ν,minimum_common_e,e1), howmanyE-minus, replace=false)
            else
                e2=[]
            end
            push!(which_Eworlds,union(minimum_common_e,e1,e2))
        end
    end
        d = zeros(ν) # we set up the probability distribution: first all worlds have zero probability
        for (i,j) in zip(which_Pworlds,which_Eworlds)
            e_and_p = intersect(j,i)
            notp=setdiff(1:ν,i)
            e_and_notp = intersect(i,notp)
            for world in e_and_p
                if world ∉ probabilizedw
                    d[world] = rand()
                    push!(probabilizedw,world)
                end
            end
            for world in notp
                if world ∉ probabilizedw
                    d[world] = rand()
                    push!(probabilizedw,world)
                end
            end
            for world in e_and_notp
                if world ∉ probabilizedw
                    d[world] = rand()
                    push!(probabilizedw,world)
                end
            end
            for world in i
                if world ∉ probabilizedw
                    d[world] = rand()
                    push!(probabilizedw,world)
                end
            end
            # we start probabilizing the worlds in d appropriately, so that our requirements may be fulfilled
            while sum(d[e_and_p])*sum(d[notp])-sum(d[e_and_notp])*sum(d[i])<=0
                # if E doesn't confirm P we have to repeat the loop
                for world in e_and_p
                    if world ∉ probabilizedw
                        d[world] = rand()
                        push!(probabilizedw,world)
                    end
                end
                for world in notp
                    if world ∉ probabilizedw
                        d[world] = rand()
                        push!(probabilizedw,world)
                    end
                end
                for world in e_and_notp
                    if world ∉ probabilizedw
                        d[world] = rand()
                        push!(probabilizedw,world)
                    end
                end
                for world in i
                    if world ∉ probabilizedw
                        d[world] = rand()
                        push!(probabilizedw,world)
                    end
                end

            end
        end

        if setdiff(1:ν,probabilizedw)!=[] # in case any of the worlds has not yet been probabilized (note: all worlds have probability >0 in simulations), we have to probabilize them too
            for world in setdiff(1:ν,probabilizedw)
                if world ∉ probabilizedw
                    d[world] = rand()
                    push!(probabilizedw,world)
                end
            end
        end

        d2=zeros(ν) # we will have to normalize distribution d as it may not necessarily sum to 1
        a0worlds=intersect(which_Pworlds...) # which worlds will determine prior probability
        normalizationquot=sum(d[a0worlds])/a0
        for world in a0worlds
            d2[world]=d[world]/normalizationquot
        end
        # we normalize prior probability (a0-worlds) to a0
        nota0worlds=setdiff(1:ν,a0worlds)
        normalizationquot=sum(d[nota0worlds])/(1-a0)
        for world in nota0worlds
            d2[world]=d[world]/normalizationquot
        end
        # then we also normalize the remaining worlds
        τ   = sample(∩(which_Eworlds...)) # pick the actual world from E-worlds
        trueprop=0
        for prop in which_Pworlds
            if issubset(τ,prop)
                trueprop+=1
            end
        end
        # count how many pieces of information are true (contain the actual E-world)

        ol1 = ols_var(d2,which_Pworlds...)
        ol2 = ols2_var(d2,which_Pworlds...)
        ol3 = olssharp_var(d2,which_Pworlds...)
        ol4 = olsstar_var(d2,which_Pworlds...)
        # calculate overlap measures (ol1: standard Olsson-Glass, ol2: our generalization, ol3: Meijs' 2005 generalisation, ol4: Koscholke et al 2019 generalization)
        return trueprop,ol1,ol2,ol3,ol4
    end

@everywhere function confSim_var3(trueenough::Int64,trueprop::Int64,ol1::Float64,ol2::Float64,ol3::Float64,ol4::Float64)
    # here we simply transform the number of true pieces of information into a categorical true (enough)/false (not true enough)
    if trueprop>=trueenough
        b=true
    else
        b=false
    end
    return b,ol1,ol2,ol3,ol4
end


@everywhere function lrMod(i::Int64, df::DataFrame)
# logistic regression: we do it in R
    dfn   = DataFrame(DV = df[!, 1], IV = df[!, i + 1])
    @rput dfn
    R"""
    m <- suppressWarnings(glm(DV ~ scale(IV), family = binomial(link = "logit"), na.action = na.exclude, data = dfn))
    prob <- predict(m, type=c('response'))
    pred <- prediction(prob, dfn$DV)
    auc <- performance(pred, 'auc')@y.values
    """
    return @rget auc
end

@everywhere function sims(n_worlds::Int64,a0::Float64,n_prop::Int64=3,trueEnoughProps::Array{Int64,1}=[3,4]) #
    res   = [ confSim_var_pre(n_worlds,a0,n_prop) for _ in 1:1000]
    df_cf = DataFrame(res)
    dfs=[]
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

function run_sim(numb_sim::Int64,a0::Float64,n_prop::Int64)
    trueEnoughProps=Array{Int64,1}(undef,0)
    for n in 1:n_prop
        if n/n_prop>=.5
            push!(trueEnoughProps,n)
        end
    end
    # for trueEnough threshold we look at at all cases with at least half pieces of information in a set presenting the true enough threshold
    outars1=Array{Any,3}(undef,length(trueEnoughProps),20,numb_sim)
    outars=[]
    for i in 1:numb_sim
        println(i)
        outars1[:,:,i] = @distributed (hcat) for n in 5:5:100
            sims(n,a0,n_prop,trueEnoughProps)
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
    titula="True enough pieces of info: "*string(trueenoughprops)*" (of "*string(n_prop)*"), a0="*string(a0)
    trace1  = scatter(x=5:5:100, y=out_auc[1, :], mode="markers+lines", name="OG",marker=attr(symbol=0),line=attr(dash="dot"))
    trace2  = scatter(x=5:5:100, y=out_auc[2, :], mode="markers+lines", name="OG-new",marker=attr(symbol=1))
    trace3  = scatter(x=5:5:100, y=out_auc[3, :], mode="markers+lines", name="OG'",marker=attr(symbol=2),line=attr(dash="dashdot"))
    trace4  = scatter(x=5:5:100, y=out_auc[4, :], mode="markers+lines", name="OG*",marker=attr(symbol=3),line=attr(dash="dash"))
    layout  = Layout(width=850, height=510, margin=attr(l=80, r=10, t=50, b=70),
                     xaxis=attr(title="Number of worlds", tickfont=attr(size=18),range=[range1,100]), yaxis=attr(tickfont=attr(size=18)), font_size=20,
                     annotations=[(x=-0.15, y=.5, xref="paper", yref="paper", text="AUC", showarrow=false, textangle=-90, font=Dict(size=>21))],title=titula)
    data    = [trace1,trace3,trace4,trace2]
    a=Plot(data, layout)
    if name[1] == '.' # if we are saving plots into current working directory, obtainable by > pwd(), changeable by cd("path")
        savefig(a,name)
    end
end

# # uncomment below for a demo run that illustrates how we obtain the plots

# n_prop=3
# n_sim=10
# a0=0.3
# results=run_sim(n_sim,a0,n_prop);
# out_ars = results[1];
# trueenprop = results[2];
# out_aucs=[];
# for prop in 1:length(results[2])
#     out_auc = mean(out_ars[prop], dims=3);
#     push!(out_aucs,out_auc)
# end
# for (i,j) in zip(trueenprop,1:length(trueenprop))
#     aucPlot("./test"*string(i)*".pdf",out_aucs[j],i,n_prop,a0)
# end

# #note: plotlyJS requires that you are connected to internet while initializing the module (in case of weird errors)

# below we use a function to output data to csv, so we don't need to repeat the simulations all the time
using CSV, DataFrames
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

# Finally, we get to a function that runs over different prior probabilities (a0s), outputs AUC plots and csv files with data.
function run_sims(numb_sim::Int64,a0s::Array{Float64,1},n_prop::Int64)

    for a0 in a0s
        println("a0")
        println(a0)
        results=run_sim(numb_sim,a0,n_prop)
        out_ars = results[1]
        trueenprop = results[2]
        out_aucs=[]
        for prop in 1:length(trueenprop)
            out_auc = mean(out_ars[prop], dims=3);
            push!(out_aucs,out_auc)
        end

        for (i,j) in zip(trueenprop,1:length(trueenprop))
            name = string("./auc-n",n_prop,"-a0_", a0, "confirmed_prop_autogenerated_",string(i),"_trueenough_prop.pdf")
            aucPlot(name,out_aucs[j],i,n_prop,a0)
            arr_to_csv(out_ars[j],string("./auc-n",n_prop,"-a0_",a0, "confirmed_prop_autogenerated_",string(i),"_trueenough_propfull.csv"))
            arr_to_csv(out_aucs[j],string("./auc-n",n_prop,"-a0_",a0, "confirmed_prop_autogenerated_",string(i),"_trueenough_prop.csv"))
        end
    end
end



# # below, the script we used for the data used in the simulations (it takes a rather long time to run through)
a0s = [0.1,0.3,0.5,0.7,0.9]
num_pieces_of_information = [3,4,5,6,7]
for num in num_pieces_of_information
    println(num)
    run_sims(250,a0s,num)
end
