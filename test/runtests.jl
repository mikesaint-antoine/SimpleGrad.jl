using SimpleGrad
using Test
using Random
Random.seed!(1234)

@testset "Value tests" begin

    rand_lower = -10
    rand_upper = 10

    function random_value(lower, upper)
        return lower + (upper - lower) * rand()
    end


    delta_var = 0.000000001


    ######################################################################

    # Addition, Value + Value
    num1_val = random_value(rand_lower, rand_upper)
    num2_val = random_value(rand_lower, rand_upper)
    num1 = Value(num1_val)
    num2 = Value(num2_val)
    num3 = num1 + num2
    backward(num3)


    # Addition, numerical derivative calculation
    num3_val = num1_val + num2_val
    num3_new = (num1_val + delta_var) + num2_val
    num1_grad = (num3_new - num3_val) / delta_var
    num3_new = num1_val + (num2_val + delta_var)
    num2_grad = (num3_new - num3_val) / delta_var


    @test abs(num1.grad - num1_grad) < 1e-5
    @test abs(num2.grad - num2_grad) < 1e-5
    @test abs(num3.data - num3_val) < 1e-5


    # Addition, Value + Number
    num1.grad=0
    num3 = num1 + num2_val
    backward(num3)

    @test abs(num1.grad - num1_grad) < 1e-5
    @test abs(num3.data - num3_val) < 1e-5


    # Addition, Number + Value
    num2.grad=0
    num3 = num1_val + num2
    backward(num3)

    @test abs(num2.grad - num2_grad) < 1e-5
    @test abs(num3.data - num3_val) < 1e-5


    ######################################################################

    # Subtraction, Value - Value
    num1_val = random_value(rand_lower, rand_upper)
    num2_val = random_value(rand_lower, rand_upper)
    num1 = Value(num1_val)
    num2 = Value(num2_val)
    num3 = num1 - num2
    backward(num3)

    # Subtraction, numerical derivative calculation
    num3_val = num1_val - num2_val
    num3_new = (num1_val + delta_var) - num2_val
    num1_grad = (num3_new - num3_val) / delta_var
    num3_new = num1_val - (num2_val + delta_var)
    num2_grad = (num3_new - num3_val) / delta_var    

    @test abs(num1.grad - num1_grad) < 1e-5
    @test abs(num2.grad - num2_grad) < 1e-5
    @test abs(num3.data - num3_val) < 1e-5



    # Subtraction, Value - Number
    num1.grad=0
    num3 = num1 - num2_val
    backward(num3)

    @test abs(num1.grad - num1_grad) < 1e-5
    @test abs(num3.data - num3_val) < 1e-5


    # Subtraction, Number - Value
    num2.grad=0
    num3 = num1_val - num2
    backward(num3)

    @test abs(num2.grad - num2_grad) < 1e-5
    @test abs(num3.data - num3_val) < 1e-5


    ######################################################################

    # Multiplication, Value * Value
    num1_val = random_value(rand_lower, rand_upper)
    num2_val = random_value(rand_lower, rand_upper)
    num1 = Value(num1_val)
    num2 = Value(num2_val)
    num3 = num1 * num2
    backward(num3)

    # Multiplication, numerical derivative calculation
    num3_val = num1_val * num2_val
    num3_new = (num1_val + delta_var) * num2_val
    num1_grad = (num3_new - num3_val) / delta_var
    num3_new = num1_val * (num2_val + delta_var)
    num2_grad = (num3_new - num3_val) / delta_var    

    @test abs(num1.grad - num1_grad) < 1e-5
    @test abs(num2.grad - num2_grad) < 1e-5
    @test abs(num3.data - num3_val) < 1e-5


    # Multiplication, Value * Number
    num1.grad=0
    num3 = num1 * num2_val
    backward(num3)

    @test abs(num1.grad - num1_grad) < 1e-5
    @test abs(num3.data - num3_val) < 1e-5


    # Multiplication, Number * Value
    num2.grad=0
    num3 = num1_val * num2
    backward(num3)

    @test abs(num2.grad - num2_grad) < 1e-5
    @test abs(num3.data - num3_val) < 1e-5


    ######################################################################

    # Division, Value / Value
    num1_val = random_value(rand_lower, rand_upper)
    num2_val = random_value(rand_lower, rand_upper)
    num1 = Value(num1_val)
    num2 = Value(num2_val)
    num3 = num1 / num2
    backward(num3)

    # Division, numerical derivative calculation
    num3_val = num1_val / num2_val
    num3_new = (num1_val + delta_var) / num2_val
    num1_grad = (num3_new - num3_val) / delta_var
    num3_new = num1_val / (num2_val + delta_var)
    num2_grad = (num3_new - num3_val) / delta_var    

    @test abs(num1.grad - num1_grad) < 1e-5
    @test abs(num2.grad - num2_grad) < 1e-5
    @test abs(num3.data - num3_val) < 1e-5


    # Division, Value / Number
    num1.grad=0
    num3 = num1 / num2_val
    backward(num3)

    @test abs(num1.grad - num1_grad) < 1e-5
    @test abs(num3.data - num3_val) < 1e-5


    # Division, Number / Value
    num2.grad=0
    num3 = num1_val / num2
    backward(num3)

    @test abs(num2.grad - num2_grad) < 1e-5
    @test abs(num3.data - num3_val) < 1e-5


    ######################################################################

    # e^x, exp(Value)
    num1_val = random_value(rand_lower, rand_upper)
    num1 = Value(num1_val)
    num2 = exp(num1)
    backward(num2)

    # e^x, numerical derivative calculation
    num2_val = exp(num1_val)
    num2_new = exp(num1_val + delta_var)
    num1_grad = (num2_new - num2_val) / delta_var

    @test abs(num1.grad - num1_grad) < 1e-5
    @test abs(num2.data - num2_val) < 1e-5

    ######################################################################

    # natural log, log(Value)
    num1_val = random_value(0, rand_upper)
    num1 = Value(num1_val)
    num2 = log(num1)
    backward(num2)

    # natural log, numerical derivative calculation
    num2_val = log(num1_val)
    num2_new = log(num1_val + delta_var)
    num1_grad = (num2_new - num2_val) / delta_var

    @test abs(num1.grad - num1_grad) < 1e-5
    @test abs(num2.data - num2_val) < 1e-5


    ######################################################################

    # x^c, Value ^ Integer
    num1_val = random_value(rand_lower, rand_upper)
    num1 = Value(num1_val)
    num2 = 2
    num3 = num1 ^ num2
    backward(num3)

    # x^c, numerical derivative calculation
    num3_val = num1_val ^ num2
    num3_new = (num1_val + delta_var) ^ num2
    num1_grad = (num3_new - num3_val) / delta_var
 

    @test abs(num1.grad - num1_grad) < 1e-5
    @test abs(num3.data - num3_val) < 1e-5


    ######################################################################

    # tanh, tanh(Value)
    num1_val = random_value(rand_lower, rand_upper)
    num1 = Value(num1_val)
    num2 = tanh(num1)
    backward(num2)

    # tanh, numerical derivative calculation
    num2_val = tanh(num1_val)
    num2_new = tanh(num1_val + delta_var)
    num1_grad = (num2_new - num2_val) / delta_var

    @test abs(num1.grad - num1_grad) < 1e-5
    @test abs(num2.data - num2_val) < 1e-5

end










@testset "Tensor tests" begin


    inputs_vals = rand(2, 3) * 10
    weights1_vals = rand(3, 4)
    weights2_vals = rand(4, 5)
    biases1_vals = rand(4)
    biases2_vals = rand(5) 
    
    
    global inputs = Tensor(inputs_vals); # Matrix with shape (2,3) -- 2 batches, 3 input features per batch
    global weights1 = Tensor(weights1_vals); # Matrix with shape (3,4) -- takes 3 inputs, has 4 neurons
    global weights2 = Tensor(weights2_vals); # Matrix with shape (4,5) -- takes 4 inputs, has 5 neurons
    global biases1 = Tensor(biases1_vals); # Bias vector for first layer neurons
    global biases2 = Tensor(biases2_vals); # Bias vector for second layer neurons
    
    layer1_out = SimpleGrad.relu(inputs * weights1 + biases1);
    layer2_out = layer1_out * weights2 + biases2;
    
    
    y_true = [0 1 0 0 0;
              0 0 0 1 0]
    
    
    loss = softmax_crossentropy(layer2_out,y_true)
    
    backward(loss)
    orig_loss = copy(loss.data)
    
    ###################################################################################################
    delta_var = 0.00000001
    
    weights1_num_grad = zeros(size(weights1))
    for i in 1:size(weights1_num_grad)[1]
        for j in 1:size(weights1_num_grad)[2]
            weights1_new = deepcopy(weights1)
            weights1_new.data[i,j] += delta_var
            layer1 = SimpleGrad.relu(inputs * weights1_new + biases1);
            layer2 = layer1 * weights2 + biases2;
            loss_new = softmax_crossentropy(layer2,y_true)
            delta_loss = loss_new.data - orig_loss
            derivative = delta_loss[1] / delta_var[1]
            weights1_num_grad[i,j] = derivative
        end
    end
    
    
    weights2_num_grad = zeros(size(weights2))
    for i in 1:size(weights2_num_grad)[1]
        for j in 1:size(weights2_num_grad)[2]
            weights2_new = deepcopy(weights2)
            weights2_new.data[i,j] += delta_var
            layer1 = SimpleGrad.relu(inputs * weights1 + biases1);
            layer2 = layer1 * weights2_new + biases2;
            loss_new = softmax_crossentropy(layer2,y_true)
            delta_loss = loss_new.data - orig_loss
            derivative = delta_loss[1] / delta_var[1]
            weights2_num_grad[i,j] = derivative
        end
    end
    
    biases1_num_grad = zeros(size(biases1))
    for i in 1:size(biases1_num_grad)[1]
        for j in 1:size(biases1_num_grad)[2]
            biases1_new = deepcopy(biases1)
            biases1_new.data[i,j] += delta_var
            layer1 = SimpleGrad.relu(inputs * weights1 + biases1_new);
            layer2 = layer1 * weights2 + biases2;
            loss_new = softmax_crossentropy(layer2,y_true)
            delta_loss = loss_new.data - orig_loss
            derivative = delta_loss[1] / delta_var[1]
            biases1_num_grad[i,j] = derivative
        end
    end
    
    biases2_num_grad = zeros(size(biases2))
    for i in 1:size(biases2_num_grad)[1]
        for j in 1:size(biases2_num_grad)[2]
            biases2_new = deepcopy(biases2)
            biases2_new.data[i,j] += delta_var
            layer1 = SimpleGrad.relu(inputs * weights1 + biases1);
            layer2 = layer1 * weights2 + biases2_new;
            loss_new = softmax_crossentropy(layer2,y_true)
            delta_loss = loss_new.data - orig_loss
            derivative = delta_loss[1] / delta_var[1]
            biases2_num_grad[i,j] = derivative
        end
    end
    




    @test all(abs.(weights1.grad .- weights1_num_grad) .< 1e-5)
    @test all(abs.(weights2.grad .- weights2_num_grad) .< 1e-5)
    @test all(abs.(biases1.grad .- biases1_num_grad) .< 1e-5)
    @test all(abs.(biases2.grad .- biases2_num_grad) .< 1e-5)

end
