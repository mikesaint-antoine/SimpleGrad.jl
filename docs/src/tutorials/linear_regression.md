# Linear Regression Example

Let's try using the *Value* class to fit a line to data, using gradient descent.

First, we'll make up some fake data of two things that are linearly related: height and basketball-skills.

```julia
using Random
Random.seed!(1234)

heights = Float64[]
for count in 1:79
    push!(heights, rand(40:90))
end


# TRUE PARAMS y = m*x + b
m = 2
b = 10



skills = Float64[]

for height in heights
    skill = m * height + b + randn() * 7.0
    push!(skills, skill)
end
```

Just for fun, I'll add myself to this dataset. I'm 72 inches tall, and extremely bad at basketball lol

```julia
push!(heights, 72)
push!(skills, 75)
```

Now let's plot the data just to take a look at it:

```julia
using Plots
scatter(heights, skills, legend=false, markersize=3, color=:black, xlabel="Height (inches)", ylabel="Basketball Skills",dpi=300)
```

![scatter_plot](../assets/plots/scatter_plot.png)


Ok, now let's see if we can use the *Value* class to fit a line to this data.

```julia
heights = [Value(item) for item in heights]
skills = [Value(item) for item in skills]
```


```julia
lr = 0.000002
runs = 100000

# initial guesses to start with
m_guess = Value(0)
b_guess = Value(0)
```

```julia
for run in 1:runs

    m_guess.grad = 0
    b_guess.grad = 0

    global loss = Value(0)

    for i in 1:length(heights)

        skill_pred = heights[i] * m_guess + b_guess
        loss_to_add = (skill_pred - skills[i])^2
        global loss += loss_to_add
    end

    
    backward(loss)

    m_guess.data -= m_guess.grad * lr
    b_guess.data -= b_guess.grad * lr

end
```

Let's see where our guesses for *m* and *b* are at now.

```julia
println(m_guess)
# output: Value(1.9906384976156302)

println(b_guess)
# output: Value(10.133894222774007)
```

Pretty close to the real values that we originally used to make the data!

We can also plot the fit line with these *m* and *b* parameters.

```julia
heights_data = [item.data for item in heights] # remember heights is full of Values, so need to do this to get the numbers

x_line = minimum(heights_data):maximum(heights_data)
y_line = m_guess.data * x_line .+ b_guess.data
plot!(x_line, y_line, linewidth=2, color=:blue)
```

![scatter_plot](../assets/plots/line_fit.png)

