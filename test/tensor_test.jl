
using SimpleGrad
using Flux
# using Flux.param
using Test


inputs_vals = rand(2, 3) * 10
weights1_vals = rand(3, 4)
weights2_vals = rand(4, 5)
biases1_vals = rand(4)
biases2_vals = rand(5) 


inputs = Tensor(inputs_vals); # Matrix with shape (2,3) -- 2 batches, 3 input features per batch
weights1 = Tensor(weights1_vals); # Matrix with shape (3,4) -- takes 3 inputs, has 4 neurons
weights2 = Tensor(weights2_vals); # Matrix with shape (4,5) -- takes 4 inputs, has 5 neurons
biases1 = Tensor(biases1_vals); # Bias vector for first layer neurons
biases2 = Tensor(biases2_vals); # Bias vector for second layer neurons

layer1_out = SimpleGrad.relu(inputs * weights1 + biases1);
layer2_out = layer1_out * weights2 + biases2;


# important -- correct classes should be one-hot encoded and NOT a Tensor, just a regular matrix.
y_true = [0 1 0 0 0;
          0 0 0 1 0]


loss = softmax_crossentropy(layer2_out,y_true)

backward(loss)



# println(weights1.grad)
# println(weights2.grad)
# println(biases1.grad)
# println(biases2.grad)

# println(loss)


##########################################################################################


# Initialize inputs, weights, and biases
# inputs_vals = rand(2, 3)
# weights1_vals = rand(3, 4)
# weights2_vals = rand(4, 5)
# biases1_vals = rand(4) * 10
# biases2_vals = rand(5) * 10

# # Define true labels (one-hot encoded)
y_true = Flux.onehotbatch([2, 4], 1:5)

# # Convert inputs and weights to Flux tensors
# inputs = Flux.params(inputs_vals)
# weights1 = Flux.params(weights1_vals)
# weights2 = Flux.params(weights2_vals)
# biases1 = Flux.params(biases1_vals)
# biases2 = Flux.params(biases2_vals)



# inputs = inputs_vals
# weights1 = weights1_vals
# weights2 = weights2_vals
# biases1 = biases1_vals
# biases2 = biases2_vals


# println(y_true)
# println(size(y_true))



test1 = SimpleGrad.relu(inputs * weights1 + biases1)
test1 = test1 * weights2 + biases2
y_true1 = [0 1 0 0 0;
          0 0 0 1 0]


exp_values = exp.(test1.data .- maximum(test1.data, dims=2))

# exp_values = exp.(test1.data )

println(maximum(test1.data, dims=2))

exit(1)
probs = exp_values ./ sum(exp_values, dims=2)

## crossentropy - sample losses
# samples = size(probs, 1)
# probs_clipped = clamp.(probs, 1e-7, 1 - 1e-7)


println(probs)
println()

# loss1 = softmax_crossentropy(test1,y_true1)


# println(loss1)
# println()

test2 = max.(0, inputs_vals * weights1_vals .+ biases1_vals')
test2 = test2 * weights2_vals .+ biases2_vals'
y_true2 = Flux.onehotbatch([2, 4], 1:5)

probs = Flux.softmax(test2)

println(probs)

# loss2 = Flux.crossentropy(Flux.softmax(test2'), y_true2)

# println(loss2)



# println(inputs)

exit(1)


# # Forward pass function
function forward_pass(inputs_vals, weights1_vals, biases1_vals, weights2_vals, biases2_vals)
    layer1_out = max.(0, inputs_vals * weights1_vals .+ biases1_vals')
    layer2_out = layer1_out * weights2_vals .+ biases2_vals'
    return layer2_out
end

# # Loss function
function compute_loss(layer2_out, y_true)
    return Flux.crossentropy(Flux.softmax(layer2_out), y_true)
end

# # Collect parameters
ps = Flux.params(inputs_vals,weights1_vals, biases1_vals, weights2_vals, biases2_vals)

# # Compute the loss and gradients
loss, back = Flux.withgradient(ps) do
    layer2_out = forward_pass(inputs_vals, weights1_vals, biases1_vals, weights2_vals, biases2_vals)
    compute_loss(layer2_out, y_true)
end

# # Extract gradients
grad_weights1 = back[weights1_vals]
grad_weights2 = back[weights2_vals]
grad_biases1 = back[biases1_vals]
grad_biases2 = back[biases2_vals]

# Print the gradients
# println("Weights 1 gradients:")
# println(grad_weights1)
# println("Weights 2 gradients:")
# println(grad_weights2)
# println("Biases 1 gradients:")
# println(grad_biases1)
# println("Biases 2 gradients:")
# println(grad_biases2)

# println("done")



println(loss)

# @testset "Tensor tests" begin


#     @test all(abs.(weights1.grad .- grad_weights1) .< 1e-6)
#     # @test all(abs.(num2.grad .- grad[2]) .< 1e-6)
#     # @test all(abs.(num3.data .- (num1_data * num2_data)) .< 1e-6)

# end


# println(weights1.grad)

println()
# println(grad_weights1)