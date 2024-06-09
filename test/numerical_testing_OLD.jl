using SimpleGrad
using Test

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

weights1_num_grad = zeros(size(weights1.grad))
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


weights2_num_grad = zeros(size(weights2.grad))
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

biases1_num_grad = zeros(size(biases1.grad))
for i in 1:size(biases1_num_grad)[1]
    biases1_new = deepcopy(biases1)
    biases1_new.data[i] += delta_var
    layer1 = SimpleGrad.relu(inputs * weights1 + biases1_new);
    layer2 = layer1 * weights2 + biases2;
    loss_new = softmax_crossentropy(layer2,y_true)
    delta_loss = loss_new.data - orig_loss
    derivative = delta_loss[1] / delta_var[1]
    biases1_num_grad[i] = derivative
end

biases2_num_grad = zeros(size(biases2.grad))
for i in 1:size(biases2_num_grad)[1]
    biases2_new = deepcopy(biases2)
    biases2_new.data[i] += delta_var
    layer1 = SimpleGrad.relu(inputs * weights1 + biases1);
    layer2 = layer1 * weights2 + biases2_new;
    loss_new = softmax_crossentropy(layer2,y_true)
    delta_loss = loss_new.data - orig_loss
    derivative = delta_loss[1] / delta_var[1]
    biases2_num_grad[i] = derivative
end






@testset "Tensor tests" begin


    @test all(abs.(weights1.grad .- weights1_num_grad) .< 1e-6)
    @test all(abs.(weights2.grad .- weights2_num_grad) .< 1e-6)
    @test all(abs.(biases1.grad .- biases1_num_grad) .< 1e-6)
    @test all(abs.(biases2.grad .- biases2_num_grad) .< 1e-6)


    # @test all(abs.(weights1.grad .- grad_weights1) .< 1e-6)
    # @test all(abs.(num2.grad .- grad[2]) .< 1e-6)
    # @test all(abs.(num3.data .- (num1_data * num2_data)) .< 1e-6)

end



println(biases1.grad)

# weights1_new = deepcopy(weights1)


# println(weights1_new)

# println("testing")