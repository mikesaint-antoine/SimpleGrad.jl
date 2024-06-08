
using SimpleGrad
using Flux
# using Flux.param



inputs_vals = rand(2, 3)
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



println(weights1.grad)
println(weights2.grad)
println(biases1.grad)
println(biases2.grad)




##########################################################################################


# Initialize inputs, weights, and biases
inputs_vals = rand(2, 3)
weights1_vals = rand(3, 4)
weights2_vals = rand(4, 5)
biases1_vals = rand(4)
biases2_vals = rand(5)

# # Define true labels (one-hot encoded)
y_true = Flux.onehotbatch([2, 4], 1:5)

# # Convert inputs and weights to Flux tensors
inputs = Flux.params(inputs_vals)
weights1 = Flux.params(weights1_vals)
weights2 = Flux.params(weights2_vals)
biases1 = Flux.params(biases1_vals)
biases2 = Flux.params(biases2_vals)

# # Forward pass function
function forward_pass(inputs, weights1, biases1, weights2, biases2)
    layer1_out = max.(0, inputs * weights1 .+ biases1')
    layer2_out = layer1_out * weights2 .+ biases2'
    return layer2_out
end

# # Loss function
function compute_loss(layer2_out, y_true)
    return Flux.crossentropy(Flux.softmax(layer2_out), y_true)
end

# # Collect parameters
ps = params(inputs,weights1, biases1, weights2, biases2)

# # Compute the loss and gradients
# loss, back = Flux.withgradient(ps) do
#     layer2_out = forward_pass(inputs, weights1, biases1, weights2, biases2)
#     compute_loss(layer2_out, y_true)
# end

# # Extract gradients
# grad_weights1 = back[weights1]
# grad_weights2 = back[weights2]
# grad_biases1 = back[biases1]
# grad_biases2 = back[biases2]

# # Print the gradients
# println("Weights 1 gradients:")
# println(grad_weights1)
# println("Weights 2 gradients:")
# println(grad_weights2)
# println("Biases 1 gradients:")
# println(grad_biases1)
# println("Biases 2 gradients:")
# println(grad_biases2)

# println("done")
