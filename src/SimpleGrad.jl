module SimpleGrad
export Value, Tensor, relu, softmax_crossentropy, backward


###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
## Value 


struct Operation{FuncType,ArgTypes}
    op::FuncType
    args::ArgTypes
end

mutable struct Value{opType}
    data::Float64
    grad::Float64
    op::opType
end

# constructor -- Value(data, grad, op)
Value(x::Number) = Value(Float64(x), 0.0, nothing)

## defines what happens when we print out a Value
import Base: show
function show(io::IO, value::Value)
    print(io, "Value(",value.data, ")")
end

backprop!(val::Value{Nothing}) = nothing


###############################################################################################################################################
## addition

import Base.+
function +(a::Value, b::Value)
    out = a.data + b.data

    # Value(data, grad, op)
    result = Value(out, 0.0, Operation(+, (a, b) ))

    return result
end

# addition for Value + number
function +(a::Value, b::Number)
    b_value = Value(Float64(b)) # cast b to Value
    return a + b_value  # use the existing method for Value + Value
end

# addition for number + Value
function +(a::Number, b::Value)
    return b + a # use Value + Number, which then casts the number to Value and does Value + Value
end





# backprop for addition operation
function backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(+), ArgTypes}

    val.op.args[1].grad += val.grad

    val.op.args[2].grad += val.grad

end



###############################################################################################################################################
## multiplication

import Base.*
function *(a::Value, b::Value)
    out = a.data * b.data

    # Value(data, grad, op)
    result = Value(out, 0.0, Operation(*, (a, b) ))

    return result
end


# Value * number
function *(a::Value, b::Number)
    b_value = Value(Float64(b))  # cast b to Value
    return a * b_value # use the existing method for Value * Value
end

# number * Value
function *(a::Number, b::Value)
    return b * a # use Value * Number, which then casts the number to Value and does Value * Value
end



# backprop for multiplication operation
function backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(*), ArgTypes}

    val.op.args[1].grad += val.op.args[2].data * val.grad

    val.op.args[2].grad += val.op.args[1].data * val.grad

end



###############################################################################################################################################
## exp


# e^x
import Base.exp
function exp(a::Value)

    out = exp(a.data)

    # Value(data, grad, op)
    result = Value(out, 0.0, Operation(exp, (a,) ))

    return result

end

# backprop for exp operation
function backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(exp), ArgTypes}

    val.op.args[1].grad += val.data * val.grad

end

###############################################################################################################################################
## log


import Base.log
function log(a::Value)

    out = log(a.data)

    # Value(data, grad, op)
    result = Value(out, 0.0, Operation(log, (a,) ))

    return result

end

# backprop for log operation
function backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(log), ArgTypes}

    val.op.args[1].grad += (1.0 / val.op.args[1].data) * val.grad

end




###############################################################################################################################################
## subtraction and negation


import Base.-

# negation
function -(a::Value)
    return a * -1
end

# subtraction: Value - Value
function -(a::Value, b::Value)
    return a + (-b)
end

# Value - number
function -(a::Value, b::Number)
    b_value = Value(Float64(b))
    return a - b_value
end


# number - Value
function -(a::Number, b::Value)
    return b - a
end


###############################################################################################################################################
# x^c


import Base.^
function ^(a::Value, b::Number)
    out = a.data ^ b

    # Value(data, grad, op)
    result = Value(out, 0.0, Operation(^, (a,b) ))

    return result
end

function backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(^), ArgTypes}

    val.op.args[1].grad += val.op.args[2] * (val.op.args[1].data ^ (val.op.args[2] - 1)) * val.grad

end


###############################################################################################################################################
# need special case for x^(-1) in Julia

import Base.inv
function inv(a::Value)
    out = 1.0 / a.data

    # Value(data, grad, op)
    result = Value(out, 0.0, Operation(inv, (a,) ))

    return result
end

function backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(inv), ArgTypes}

    val.op.args[1].grad -= (1.0 / (val.op.args[1].data * val.op.args[1].data)) * val.grad

end


###############################################################################################################################################
# division: a / b = a * b^(-1)

# for now both must be Values
# TODO -- implement Value / number and number / Value at some point? 

import Base./
function /(a::Value, b::Value)
    return a * (b ^ -1)
end


###############################################################################################################################################
# tanh

import Base.tanh
function tanh(a::Value)

    out = (exp(2 * a.data) - 1) / (exp(2 * a.data) + 1)

    # Value(data, grad, op)
    result = Value(out, 0.0, Operation(tanh, (a,) ))

    return result
end


function backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(tanh), ArgTypes}

    val.op.args[1].grad += (1 - val.data^2) * val.grad

end


###############################################################################################################################################
## full backward pass

function backward(a::Value)

    function build_topo(v::Value, visited=Value[], topo=Value[])
        if !(v in visited)
            push!(visited, v)

            if v.op != nothing
                for operand in v.op.args
                    if operand isa Value
                        build_topo(operand, visited, topo)
                    end
                end
            end

            push!(topo, v)
        end
        return topo
    end
    
    topo = build_topo(a)

    a.grad = 1.0
    for node in reverse(topo)
        backprop!(node)
    end

end

###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
## Tensor 



mutable struct Tensor{opType}
    data::Union{Array{Float64,1},Array{Float64,2}}
    grad::Union{Array{Float64,1},Array{Float64,2}}
    op::opType
end


# constructor -- Tensor(data, grad, op)
Tensor(x::Union{Array{Float64,1},Array{Float64,2}}) = Tensor(x, zeros(Float64, size(x)), nothing)



import Base.show
function show(io::IO, tensor::Tensor)
    print(io, "Tensor(",tensor.data, ")")
end


backprop!(tensor::Tensor{Nothing}) = nothing

###############################################################################################################################################

# multiplication



import Base.*
function *(a::Tensor, b::Tensor)

    out = a.data * b.data

    # Tensor(data, grad, op)
    result = Tensor(out, zeros(Float64, size(out)), Operation(*, (a, b)))

    return result
end



function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(*), ArgTypes}

    tensor.op.args[1].grad += tensor.grad * transpose(tensor.op.args[2].data)

    tensor.op.args[2].grad += transpose(tensor.op.args[1].data) * tensor.grad 

end

###############################################################################################################################################

# addition


# define addition for 2 Tensor objects
import Base.+
function +(a::Tensor, b::Tensor)

    if length(size(a.data)) == length(size(b.data))
        out = a.data .+ b.data
    elseif length(size(a.data)) > length(size(b.data))
        # a is 2D, b is 1D
        out = a.data .+ transpose(b.data)
    else
        # a is 1D, b is 2D
        out = b.data .+ transpose(a.data)
    end

    # Tensor(data, grad, op)
    result = Tensor(out, zeros(Float64, size(out)), Operation(+, (a, b)))

    return result
end


# backprop for addition operation
function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(+), ArgTypes}


    if length(size(tensor.grad)) > length(size(tensor.op.args[1].data))
        tensor.op.args[1].grad += dropdims(sum(tensor.grad, dims=1), dims=1) # need dropdims to make this size (x,) rather than size (1,x)
    else
        tensor.op.args[1].grad += ones(size(tensor.op.args[1].data)) .* tensor.grad
    end

    if length(size(tensor.grad)) > length(size(tensor.op.args[2].data))
        tensor.op.args[2].grad += dropdims(sum(tensor.grad, dims=1), dims=1) # need dropdims to make this size (x,) rather than size (1,x)
    else
        tensor.op.args[2].grad += ones(size(tensor.op.args[2].data)) .* tensor.grad
    end

end


###############################################################################################################################################

# relu

function relu(a::Tensor)

    out = max.(a.data,0)

    # Tensor(data, grad, op)
    result = Tensor(out, zeros(Float64, size(out)), Operation(relu, (a,)))

    return result
end




function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(relu), ArgTypes}

    tensor.op.args[1].grad += (tensor.op.args[1].data .> 0) .* tensor.grad

end


###############################################################################################################################################

# softmax_crossentropy


# TODO - y_true must be Tensor here, not in original version (used to just be array)
function softmax_crossentropy(a::Tensor,y_true::Union{Array{Int,2},Array{Float64,2}}; grad::Bool=true)

    ## implementing softmax activation and cross entropy loss separately leads to very complicated gradients
    ## but combining them makes the gradient a lot easier to deal with

    ## credit to Sendex and his textbook for teaching me this part
    ## great textbook for doing this stuff in Python, you can get it here:
    ## https://nnfs.io/

    # softmax activation
    exp_values = exp.(a.data .- maximum(a.data, dims=2))
    probs = exp_values ./ sum(exp_values, dims=2)
    
    ## crossentropy - sample losses
    samples = size(probs, 1)
    probs_clipped = clamp.(probs, 1e-7, 1 - 1e-7)
    # deal with 0s


    # basically just returns an array with the probability of the correct answer for each batch
    correct_confidences = sum(probs_clipped .* y_true, dims=2)

    # negative log likelihood
    sample_losses = -log.(correct_confidences)


    # loss_mean
    out = [mean(sample_losses)]



    if grad

        # it's easier to do the grad calculation here because doing it seperately will involve redoing a lot of calculations

        samples = size(probs, 1)

        # convert from one-hot to index list
        y_true_argmax = argmax(y_true, dims=2)

        a.grad = copy(probs)
        for samp_ind in 1:samples
            a.grad[samp_ind, y_true_argmax[samp_ind][2]] -= 1
            ## this syntax y_true_argmax[i][2] is just to get the column index of the true value
        end
        a.grad ./= samples


    end

    # Tensor(data, grad, op)
    result = Tensor(out, zeros(Float64, size(out)), Operation(softmax_crossentropy, (a,)))

    return result
end



function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(softmax_crossentropy), ArgTypes}

end








###############################################################################################################################################

# full backward pass



function backward(a::Tensor)

    function build_topo(v::Tensor, visited=Tensor[], topo=Tensor[])
        if !(v in visited)
            push!(visited, v)

            if v.op != nothing
                for operand in v.op.args
                    if operand isa Tensor
                        build_topo(operand, visited, topo)
                    end
                end
            end

            push!(topo, v)
        end
        return topo
    end
    
    topo = build_topo(a)

    a.grad .= 1.0
    for node in reverse(topo)
        backprop!(node)
    end

end




end # module
