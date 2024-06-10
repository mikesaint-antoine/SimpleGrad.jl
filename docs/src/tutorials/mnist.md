
# MNIST Example

In this section, we'll use *Tensors* for a real neural net example -- solving the [MNIST image classification problem.](https://en.wikipedia.org/wiki/MNIST_database) The idea in this problem is to use a neural net to "read" images of hand-drawn numbers, from 0-9, and correctly identify which number each one is.

However, we're gonna gonna be a bit lazy and work with pre-processed data in CSV format, rather than actually reading in the images ourselves. [You can download the MNIST data in CSV format here.](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)


First, we'll read in the training data. We want to store the features (pixel values from the images) in an array called `X`, and store the labels (the correct answers for which number each one is) in an array called `y`. Lastly, we scale the features by diving them by 255, to get them between 0 and 1. Here's the code:

```julia
X = []
y = []
global first_row = true
open("mnist_data/mnist_train.csv", "r") do file
    for line in eachline(file)

        if first_row  # skip the first row
            global first_row = false
            continue
        end

        # split the line by comma and strip whitespace
        row = parse.(Float64, strip.(split(line, ',')))

        push!(y, row[1])
        push!(X, row[2:length(row)])
    end
end

X= hcat(X...)'
X = X / 255.0
```

Next, we do the same thing for the testing data, except we save the features and labels in arrays called `X_test` and `y_test`. Here's the code:

```julia
X_test = []
y_test = []
global first_row = true
open("mnist_data/mnist_test.csv", "r") do file
    for line in eachline(file)

        if first_row  # skip the first row
            global first_row = false
            continue
        end

        # split the line by comma and strip whitespace
        row = parse.(Float64, strip.(split(line, ',')))

        push!(y_test, row[1])
        push!(X_test, row[2:length(row)])
    end
end

X_test = hcat(X_test...)'
X_test = X_test / 255.0
```


Next, we define the model...

```julia
using SimpleGrad
using Random
Random.seed!(1234)
# seeding the random number generator for reproducibility

weights1 = Tensor(0.01 * rand(784, 128))
weights2 = Tensor(0.01 * rand(128, 10))

biases1 = Tensor(zeros(128))
biases2 = Tensor(zeros(10))


batch_size = 100
num_classes = 10  # total number of classes
lr = 0.1
epochs = 2
```

Now, we train the model:

```julia
global run = 1
for epoch in 1:epochs

    for i in 1:batch_size:size(X,1)


        ## get current batch
        batch_X = X[i:i+batch_size-1, :]
        batch_X = Tensor(batch_X)
        batch_y = y[i:i+batch_size-1]



        ## convert batch_y to one-hot
        batch_y_one_hot = zeros(batch_size,num_classes)
        for batch_ind in 1:batch_size
            batch_y_one_hot[batch_ind,Int.(batch_y)[batch_ind]+1] = 1
        end



        ## zero grads
        weights1.grad .= 0
        weights2.grad .= 0
        biases1.grad .= 0
        biases2.grad .= 0



        ## forward pass
        layer1_out = relu(batch_X * weights1 + biases1)
        layer2_out = layer1_out * weights2 + biases2
        loss = softmax_crossentropy(layer2_out,batch_y_one_hot)



        ## backward pass
        backward(loss)


        ## update params
        weights1.data -= weights1.grad .* lr
        weights2.data -= weights2.grad .* lr
        biases1.data -= biases1.grad .* lr
        biases2.data -= biases2.grad .* lr


        if run % 100 == 0
            println("Epoch: $epoch, run: $run, loss: $(round(loss.data[1], digits=3))")
        end
        
        global run += 1

    end
end

# output:
# Epoch: 1, run: 100, loss: 1.145
# Epoch: 1, run: 200, loss: 0.55
# Epoch: 1, run: 300, loss: 0.677
# Epoch: 1, run: 400, loss: 0.491
# Epoch: 1, run: 500, loss: 0.331
# Epoch: 1, run: 600, loss: 0.388
# Epoch: 2, run: 700, loss: 0.225
# Epoch: 2, run: 800, loss: 0.299
# Epoch: 2, run: 900, loss: 0.502
# Epoch: 2, run: 1000, loss: 0.355
# Epoch: 2, run: 1100, loss: 0.248
# Epoch: 2, run: 1200, loss: 0.337
```


Finally, we check out performance on the testing set:

```julia
global correct = 0
global total = 0
for i in 1:length(y_test)
    X_in = X_test[i:i,:] ## need to keep this (1,784), not (784,)
    X_in = Tensor(X_in)
    y_true = y_test[i]

    layer1_out = relu(X_in * weights1 + biases1)
    layer2_out = layer1_out * weights2 + biases2


    pred_argmax = argmax(layer2_out.data, dims=2)[1][2]

    if pred_argmax-1 == y_true
        global correct +=1
    end
    global total += 1

end

println(correct/total)
# output: 0.9187
```

91.87% accuracy on the testing set. Not bad!