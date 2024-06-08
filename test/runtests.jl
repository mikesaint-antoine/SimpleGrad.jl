using SimpleGrad
using Test
using Flux

rand_lower = -20
rand_upper = 20

function random_value(lower, upper)
    return lower + (upper - lower) * rand()
end




@testset "Value tests" begin

    # Addition, Value + Value
    num1_val = random_value(rand_lower, rand_upper)
    num2_val = random_value(rand_lower, rand_upper)
    num1 = Value(num1_val)
    num2 = Value(num2_val)
    num3 = num1 + num2
    backward(num3)

    flux_add(a, b) = a + b
    grad = gradient(flux_add, num1_val, num2_val)

    @test abs(num1.grad - grad[1]) < 1e-6
    @test abs(num2.grad - grad[2]) < 1e-6
    @test abs(num3.data - (num1_val + num2_val)) < 1e-6

    # Subtraction, Value - Value
    num1_val = random_value(rand_lower, rand_upper)
    num2_val = random_value(rand_lower, rand_upper)
    num1 = Value(num1_val)
    num2 = Value(num2_val)
    num3 = num1 - num2
    backward(num3)

    flux_subtract(a, b) = a - b
    grad = gradient(flux_subtract, num1_val, num2_val)

    @test abs(num1.grad - grad[1]) < 1e-6
    @test abs(num2.grad - grad[2]) < 1e-6
    @test abs(num3.data - (num1_val - num2_val)) < 1e-6

    # Multiplication, Value * Value
    num1_val = random_value(rand_lower, rand_upper)
    num2_val = random_value(rand_lower, rand_upper)
    num1 = Value(num1_val)
    num2 = Value(num2_val)
    num3 = num1 * num2
    backward(num3)

    flux_multiply(a, b) = a * b
    grad = gradient(flux_multiply, num1_val, num2_val)

    @test abs(num1.grad - grad[1]) < 1e-6
    @test abs(num2.grad - grad[2]) < 1e-6
    @test abs(num3.data - (num1_val * num2_val)) < 1e-6

    # Division, Value / Value
    num1_val = random_value(rand_lower, rand_upper)
    num2_val = random_value(rand_lower, rand_upper)
    num1 = Value(num1_val)
    num2 = Value(num2_val)
    num3 = num1 / num2
    backward(num3)

    flux_divide(a, b) = a / b
    grad = gradient(flux_divide, num1_val, num2_val)

    @test abs(num1.grad - grad[1]) < 1e-6
    @test abs(num2.grad - grad[2]) < 1e-6
    @test abs(num3.data - (num1_val / num2_val)) < 1e-6

    # e^x, exp(Value)
    num1_val = random_value(rand_lower, rand_upper)
    num1 = Value(num1_val)
    num2 = exp(num1)
    backward(num2)

    flux_exp(a) = exp(a)
    grad = gradient(flux_exp, num1_val)

    @test abs(num1.grad - grad[1]) < 1e-6
    @test abs(num2.data - exp(num1_val)) < 1e-6

    # natural log, log(Value)
    num1_val = random_value(0, rand_upper)
    num1 = Value(num1_val)
    num2 = log(num1)
    backward(num2)

    flux_log(a) = log(a)
    grad = gradient(flux_log, num1_val)

    @test abs(num1.grad - grad[1]) < 1e-6
    @test abs(num2.data - log(num1_val)) < 1e-6

    # x^c, Value^c
    num1_val = random_value(0, rand_upper)
    num2_val = random_value(rand_lower, rand_upper) * 0.1
    num1 = Value(num1_val)
    num3 = num1 ^ num2_val
    backward(num3)

    flux_power(a, b) = a ^ b
    grad = gradient(flux_power, num1_val, num2_val)

    @test abs(num1.grad - grad[1]) < 1e-6
    @test abs(num3.data - (num1_val ^ num2_val)) < 1e-6

    # tanh, tanh(Value)
    num1_val = random_value(rand_lower, rand_upper)
    num1 = Value(num1_val)
    num2 = tanh(num1)
    backward(num2)

    flux_tanh(a) = tanh(a)
    grad = gradient(flux_tanh, num1_val)

    @test abs(num1.grad - grad[1]) < 1e-6
    @test abs(num2.data - tanh(num1_val)) < 1e-6
end





using SimpleGrad
using Test
using Flux

function random_tensor(rows, cols, lower, upper)
    return lower .+ (upper .- lower) .* rand(rows, cols)
end

@testset "Tensor tests" begin

    # Multiplication, Tensor * Tensor
    rows, cols = 2, 2
    rand_lower, rand_upper = -20, 20

    num1_data = random_tensor(rows, cols, rand_lower, rand_upper)
    num2_data = random_tensor(rows, cols, rand_lower, rand_upper)
    num1 = Tensor(num1_data)
    num2 = Tensor(num2_data)
    num3 = num1 * num2
    backprop!(num3)

    flux_multiply(a, b) = a * b
    grad = gradient(flux_multiply, num1_data, num2_data)

    @test all(abs.(num1.grad .- grad[1]) .< 1e-6)
    @test all(abs.(num2.grad .- grad[2]) .< 1e-6)
    @test all(abs.(num3.data .- (num1_data * num2_data)) .< 1e-6)

end

