module SimpleGrad
export Value, Tensor, relu, softmax_crossentropy, backward, zero_grad

# Base functions that we're going to be overriding 
import Base: ==, show, +, *, exp, log, -, ^, inv, /, tanh


# Operation
struct Operation{FuncType,ArgTypes}
    op::FuncType
    args::ArgTypes
end


# Value
mutable struct Value{opType} <: Number
    data::Float64
    grad::Float64
    op::opType
end


# constructor -- Value(data, grad, op)
Value(x::Number) = Value(Float64(x), 0.0, nothing)


Base.promote_rule(::Type{<:Value}, ::Type{T}) where {T<:Number} = Value


function ==(a::Value, b::Value)
    return a===b
end

function show(io::IO, value::Value)
    print(io, "Value(",value.data, ")")
end

backprop!(val::Value{Nothing}) = nothing



# addition
function +(a::Value, b::Value)

    out = a.data + b.data
    result = Value(out, 0.0, Operation(+, (a, b) )) # Value(data, grad, op)
    return result

end


# addition backprop
function backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(+), ArgTypes}

    val.op.args[1].grad += val.grad # update gradient of first operand
    val.op.args[2].grad += val.grad # update gradient of second operand

end


# multiplication
function *(a::Value, b::Value)

    out = a.data * b.data
    result = Value(out, 0.0, Operation(*, (a, b) )) # Value(data, grad, op)
    return result

end


# multiplication backprop
function backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(*), ArgTypes}

    val.op.args[1].grad += val.op.args[2].data * val.grad
    val.op.args[2].grad += val.op.args[1].data * val.grad

end


# e^x
function exp(a::Value)

    out = exp(a.data)
    result = Value(out, 0.0, Operation(exp, (a,) )) # Value(data, grad, op)
    return result

end

# e^x backprop
function backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(exp), ArgTypes}

    val.op.args[1].grad += val.data * val.grad

end



# log
function log(a::Value)

    out = log(a.data)
    result = Value(out, 0.0, Operation(log, (a,) )) # Value(data, grad, op)
    return result

end

# log backprop
function backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(log), ArgTypes}

    val.op.args[1].grad += (1.0 / val.op.args[1].data) * val.grad

end





# negation (no backprop needed, just turn it into multiplication)
function -(a::Value)

    return a * -1

end

# subtraction (no backprop needed, just turn it into negation and addition)
function -(a::Value, b::Value)

    return a + (-b)

end




# exponents -- for now, exponent must be Integer and can't be Value or Float
function ^(a::Value, b::Integer)

    out = a.data ^ b
    result = Value(out, 0.0, Operation(^, (a,b) )) # Value(data, grad, op)
    return result

end

function ^(a::Value, b::Number)
    throw(MethodError(^, (Value, Number), "Exponentiation with non-integer exponents is not supported."))
end

function ^(a::Number, b::Value)
    throw(MethodError(^, (Number, Value), "Exponentiation with Value as the exponent is not supported."))
end


# exponents backprop
function backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(^), ArgTypes}

    val.op.args[1].grad += val.op.args[2] * (val.op.args[1].data ^ (val.op.args[2] - 1)) * val.grad

end




# need special case for x^(-1) in Julia -- called "inv()"
function inv(a::Value)

    out = 1.0 / a.data
    result = Value(out, 0.0, Operation(inv, (a,) )) # Value(data, grad, op)
    return result

end

# inv() backprop
function backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(inv), ArgTypes}

    val.op.args[1].grad -= (1.0 / (val.op.args[1].data * val.op.args[1].data)) * val.grad

end


# division (no backprop needed, just turn it into inv() and multiplication)
# a / b = a * b^(-1)
function /(a::Value, b::Value)

    return a * inv(b)

end



# tanh
function tanh(a::Value)

    out = (exp(2 * a.data) - 1) / (exp(2 * a.data) + 1)
    result = Value(out, 0.0, Operation(tanh, (a,) )) # Value(data, grad, op)
    return result

end

# tanh backprop
function backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(tanh), ArgTypes}

    val.op.args[1].grad += (1 - val.data^2) * val.grad

end


## full backward pass for Value
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








# Tensor 
mutable struct Tensor{opType} <: AbstractArray{Float64, 2}
    data::Array{Float64, 2}
    grad::Array{Float64, 2}
    op::opType
end


# 2D constructor -- Tensor(data, grad, op)
Tensor(x::Array{Float64,2}) = Tensor(x, zeros(Float64, size(x)), nothing)


# 1D constructor - reshape to 2D (row vector or column vector) then call 2D constructor
function Tensor(x::Array{Float64, 1}; column_vector::Bool=false)

    if column_vector
        # column vector - size (N,1)
        data_2D = reshape(x, (length(x), 1))
    else
        # DEFAULT row vector - size (1,N)
        data_2D = reshape(x, (1,length(x)))
    end

    Tensor(data_2D, zeros(Float64, size(data_2D)), nothing) # Tensor(data, grad, op)

end


# original show function for Tensor. commenting out because i've now added a nicer show function (at the bottom of this file)
# function show(io::IO, tensor::Tensor)
#     print(io, "Tensor(",tensor.data, ")")
# end

backprop!(tensor::Tensor{Nothing}) = nothing


function ==(a::Tensor, b::Tensor)
    return a===b
end




Base.size(x::Tensor) = size(x.data)

Base.getindex(x::Tensor, i...) = getindex(x.data, i...)

Base.setindex!(x::Tensor, v, i...) = setindex!(x.data, v, i...)



# matrix multiplication
function *(a::Tensor, b::Tensor)

    out = a.data * b.data
    result = Tensor(out, zeros(Float64, size(out)), Operation(*, (a, b))) # Tensor(data, grad, op)
    return result

end


# matrix multiplication backprop
function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(*), ArgTypes}

    tensor.op.args[1].grad += tensor.grad * transpose(tensor.op.args[2].data)
    tensor.op.args[2].grad += transpose(tensor.op.args[1].data) * tensor.grad 

end




# matrix addition
function +(a::Tensor, b::Tensor)

    # broadcasting happens automatically for row-vector
    out = a.data .+ b.data
    result = Tensor(out, zeros(Float64, size(out)), Operation(+, (a, b))) # Tensor(data, grad, op)
    return result

end


# matrix addition backprop
function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(+), ArgTypes}

    if size(tensor.grad) == size(tensor.op.args[1].data)
        tensor.op.args[1].grad += ones(size(tensor.op.args[1].data)) .* tensor.grad
    else
        # reverse broadcast
        tensor.op.args[1].grad += ones(size(tensor.op.args[1].grad)) .* sum(tensor.grad,dims=1)
    end

    if size(tensor.grad) == size(tensor.op.args[2].data)
        tensor.op.args[2].grad += ones(size(tensor.op.args[2].data)) .* tensor.grad
    else
        # reverse broadcast
        tensor.op.args[2].grad += ones(size(tensor.op.args[2].grad)) .* sum(tensor.grad,dims=1)
    end

end




# relu
function relu(a::Tensor)

    out = max.(a.data,0)
    result = Tensor(out, zeros(Float64, size(out)), Operation(relu, (a,))) # Tensor(data, grad, op)
    return result

end



# relu backprop
function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(relu), ArgTypes}

    tensor.op.args[1].grad += (tensor.op.args[1].data .> 0) .* tensor.grad

end



# softmax_crossentropy
function softmax_crossentropy(a::Tensor,y_true::Union{Array{Int,2},Array{Float64,2}}; grad::Bool=true)

    ## implementing softmax activation and cross entropy loss separately leads to very complicated gradients
    ## but combining them makes the gradient a lot easier to deal with

    ## credit to Sendex and his textbook for teaching me this part
    ## great textbook for doing this stuff in Python, you can get it here:
    ## https://nnfs.io/

    # softmax activation
    exp_values = exp.(a.data .- maximum(a.data, dims=2))
    probs = exp_values ./ sum(exp_values, dims=2)

    probs_clipped = clamp.(probs, 1e-7, 1 - 1e-7)
    # deal with 0s and 1s


    # basically just returns an array with the probability of the correct answer for each batch
    correct_confidences = sum(probs_clipped .* y_true, dims=2)

    # negative log likelihood
    sample_losses = -log.(correct_confidences)

    # loss mean
    out = [sum(sample_losses) / length(sample_losses)]



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

    # reshape out from (1,) to (1,1) 
    out = reshape(out, (1, 1))

    result = Tensor(out, zeros(Float64, size(out)), Operation(softmax_crossentropy, (a,))) # Tensor(data, grad, op)

    return result

end


# softmax_crossentropy backprop is empty because gradient is easier to calculate during forward pass
function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(softmax_crossentropy), ArgTypes}

end



## full backward pass for Tensor
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


#############################################################################################
# NOTE: past this point is new code not in the original tutorial website or video series
#############################################################################################


# zero_grad (Value) - same idea as build_topo()
function zero_grad(v::Value, visited=Value[])
    if !(v in visited)

        push!(visited, v)
        v.grad = 0

        if v.op != nothing
            for operand in v.op.args
                if operand isa Value
                    zero_grad(operand, visited)
                end
            end
        end

    end
end


# zero_grad (Tensor)
function zero_grad(v::Tensor, visited=Tensor[])
    if !(v in visited)

        push!(visited, v)

        v.grad .= 0

        if v.op != nothing
            for operand in v.op.args
                if operand isa Tensor
                    zero_grad(operand, visited)
                end
            end
        end

    end
end

# nicer printing for Tensors, overwriting old show(io::IO, tensor::Tensor) function
function show(io::IO, tensor::Tensor)

    print(io, "Tensor(")
    for i in 1:size(tensor)[1]

        if i==1
            print(io, tensor.data[i,:], "\n")
        elseif i < size(tensor)[1]
            print(io, "       ",tensor.data[i,:], "\n")
        else
            print(io, "       ",tensor.data[i,:], ")")
        end

    end

end






end # module