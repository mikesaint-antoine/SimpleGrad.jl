
using SimpleGrad
import Statistics


# struct Operation{FuncType,ArgTypes}
#     op::FuncType
#     args::ArgTypes
# end


# import Base.*
# function *(a::Tensor, b::Tensor)

#     out = a.data * b.data

#     # Tensor(data, grad, op)
#     result = Tensor(out, zeros(Float64, size(out)), Operation(*, (a, b)))

#     return result
# end



# import Base.+
# function +(a::Tensor, b::Tensor)

#     if length(size(a.data)) == length(size(b.data))
#         out = a.data .+ b.data
#     elseif length(size(a.data)) > length(size(b.data))
#         # a is 2D, b is 1D
#         out = a.data .+ transpose(b.data)
#     else
#         # a is 1D, b is 2D
#         out = b.data .+ transpose(a.data)
#     end

#     # Tensor(data, grad, op)
#     result = Tensor(out, zeros(Float64, size(out)), Operation(+, (a, b)))

#     return result
# end



inputs = Tensor(rand(2, 3)); # Matrix with shape (2,3) -- 2 batches, 3 input features per batch
weights1 = Tensor(rand(3, 4)); # Matrix with shape (3,4) -- takes 3 inputs, has 4 neurons
weights2 = Tensor(rand( 4, 5)); # Matrix with shape (4,5) -- takes 4 inputs, has 5 neurons
biases1 = Tensor([1.0,1.0,1.0,1.0]); # Bias vector for first layer neurons
biases2 = Tensor([1.0,1.0,1.0,1.0,1.0]); # Bias vector for second layer neurons

layer1_out = relu(inputs * weights1 + biases1);
layer2_out = layer1_out * weights2 + biases2;


# important -- correct classes should be one-hot encoded and NOT a Tensor, just a regular matrix.
y_true = [0 1 0 0 0;
          0 0 0 1 0]


loss = softmax_crossentropy(layer2_out,y_true)



println(loss)

println("done")