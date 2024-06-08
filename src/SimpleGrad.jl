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









end # module
