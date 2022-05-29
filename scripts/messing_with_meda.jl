using DrWatson
@quickactivate "Messing With MEDA"
using Arrow
using Base.Threads: @threads
using DataFrames
using Flux
using Flux: mse, DataLoader, train!, params
using ProgressMeter: @showprogress
using PyPlot
using Random: shuffle
using StatsBase: mean, std, zscore

pygui(true)

## learning functions

function split(X, y, validation::Int=1, test::Int=1)
    n = size(X,2)
    @assert size(y,2) == n
    idx = shuffle(1:n)
    t = n ÷ 10
    tr = t*(10 - validation - test)
    va = t*(10 - validation)
    return(
        X[:,idx[1:tr]], #training data
        y[:,idx[1:tr]], #training labels
        X[:,idx[tr+1:va]], #validation data
        y[:,idx[tr+1:va]], #validation labels
        X[:,idx[va+1:end]], #test data
        y[:,idx[va+1:end]], #test labels 
    )
end

TOY(sol::AbstractVector) = (sin.(2π*sol/668), cos.(2π*sol/668))

TOD(sol::AbstractVector) = (sin.(2π*sol), cos.(2π*sol))

function sinusoids(sol::AbstractVector)
    toys, toyc = TOY(sol)
    tods, todc = TOD(sol)
    hcat(toys, toyc, tods, todc) |> transpose |> Matrix .|> Float32
end

function train(nlayer, hlayer, Xtr, ytr, Xva, yva, Xte, yte, nepoch=8, nbatch=256)
    @assert nlayer >= 3
    #construct hidden layers as specified
    hidden = fill(Dense(hlayer => hlayer, relu), nlayer - 2)
    println(hidden)
    #construct a model
    model = Chain(
        Dense(size(X,1)=>hlayer, relu),
        hidden...,
        Dense(hlayer => 1)
    )
    #define the loss function
    loss(x, y) = mse(model(x), y)
    loss(z) = loss(z...)
    #define training parameters
    dl = DataLoader((data=Xtr, labels=ytr), batchsize=nbatch)
    p = params(model)
    opt = ADAM()
    #train the model
    history = (tr_loss=zeros(nepoch+1), va_loss=zeros(nepoch+1))
    history[:tr_loss][1] = loss(Xtr, ytr)
    history[:va_loss][1] = loss(Xva, yva)
    @showprogress for i ∈ 1:nepoch
        train!(loss, p, dl, opt)
        history[:tr_loss][i+1] = loss(Xtr, ytr)
        history[:va_loss][i+1] = loss(Xva, yva)
    end
    return(model, history, loss(Xte, yte))
end

## load the data

odf = datadir("pro", "meda.feather") |> Arrow.Table |> DataFrame

## process the data

#average everything within a single minute, cutting down noise and size
df = combine(
    groupby(
        odf,
        [:sol, :hr, :min]
    ),
    names(odf) .=> mean .=> names(odf)
)
#fold all the time information into fractional sols
df.sol = df.sol + df.hr/24 + df.min/(24*60) + df.sec/(24*3600)
df.sol .-= minimum(df.sol)
#get rid of useless columns
rename!(df, :pressure => :P)
select!(df, :sol, :P)
dropmissing!(df)

## sinusoidal representations of time of year and time of day

df[!,:toys], df[!,:toyc] = TOY(df.sol)
df[!,:tods], df[!,:todc] = TOD(df.sol)

## prepare data for modeling

#prediction columns
pred = [:toys, :toyc, :tods, :todc]
#target/label column
targ = :P #pressure

#convert into nice matrices
X = df[!,pred] |> Matrix |> transpose |> Matrix .|> Float32
y = df[!,targ] |> zscore |> transpose |> Matrix .|> Float32
μ = mean(df.P)
σ = std(df.P)

#training, validation, and test groups of data
Xtr, ytr, Xva, yva, Xte, yte = split(X, y)

##

nlayer = [4, 5, 6, 8]
hlayer = [8, 16, 32, 64]
figa, axsa = subplots(
    length(hlayer),
    length(nlayer),
    figsize=(6.5,6.5),
    sharex=true,
    sharey=true,
    constrained_layout=true
)
figb, axsb = subplots(
    length(hlayer),
    length(nlayer),
    figsize=(6.5,6.5),
    sharex=true,
    sharey=true,
    constrained_layout=true
)
sola = LinRange(0, 668*2, 1_000_000)
solb = LinRange(100, 102, 100_000)
idxb = 100 .< df.sol .< 102
for i ∈ 1:length(hlayer)
    for j ∈ 1:length(nlayer)
        model, history, te_loss = train(nlayer[j], hlayer[i], Xtr, ytr, Xva, yva, Xte, yte)
        axsa[i,j].plot(df.sol, df.P, color="k", alpha=0.5)
        axsa[i,j].plot(sola, μ .+ σ*(sinusoids(sola) |> model |> vec), alpha=0.7)
        axsb[i,j].plot(df.sol[idxb], df.P[idxb], color="k", alpha=0.5)
        axsb[i,j].plot(solb, μ .+ σ*(sinusoids(solb) |> model |> vec), alpha=0.7)
    end
end
for i ∈ 1:length(hlayer)
    axsa[i,1].set_ylabel("Layer Size = $(hlayer[i])")
    axsb[i,1].set_ylabel("Layer Size = $(hlayer[i])")
end
for j ∈ 1:length(nlayer)
    axsa[1,j].set_title("$(nlayer[j]) Layers")
    axsb[1,j].set_title("$(nlayer[j]) Layers")
end
figa.supxlabel("Time [sol]")
figb.supxlabel("Time [sol]")
figa.supylabel("Pressure [Pa]")
figb.supylabel("Pressure [Pa]")
figa.savefig(plotsdir("model_comparison_long"), dpi=500)
figb.savefig(plotsdir("model_comparison_short"), dpi=500)