
## MNIST Example

Lastly, let's try out a real neural net example -- solving the MNIST image classification problem.

You can download the MNIST data in CSV format here:
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

First, we'll read in the training and testing data...

```julia
## read training data

X = []
y = []
global first_row = true
open("mnist_data/mnist_train.csv", "r") do file
    for line in eachline(file)

        if first_row  # Skip the first row
            global first_row = false
            continue
        end

        # Split the line by comma and strip whitespace
        row = parse.(Float64, strip.(split(line, ',')))

        push!(y, row[1])
        push!(X, row[2:length(row)])
    end
end

X= hcat(X...)';
X = X / 255.0;



## read testing data

X_test = []
y_test = []
global first_row = true
open("mnist_data/mnist_test.csv", "r") do file
    for line in eachline(file)

        if first_row  # Skip the first row
            global first_row = false
            continue
        end

        # Split the line by comma and strip whitespace
        row = parse.(Float64, strip.(split(line, ',')))

        push!(y_test, row[1])
        push!(X_test, row[2:length(row)])
    end
end

X_test = hcat(X_test...)';
X_test = X_test / 255.0;
```


Next, we define the model...

```julia
## define model

weights1 = Tensor(0.01 * rand(784, 128));
weights2 = Tensor(0.01 * rand(128, 10));

biases1 = Tensor(zeros(128));
biases2 = Tensor(zeros(10));


batch_size = 100;
num_classes = 10;  # total number of classes
lr = 0.1;
epochs = 2;
```

Now, we train the model...


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


        if run % 10 == 0
            println("Epoch: $epoch, run: $run, loss: $(round(loss.data[1], digits=3))")
        end
        
        global run += 1

    end
end
```



Finally, we check out performance on the testing set...

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
```


