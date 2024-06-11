using Documenter
using SimpleGrad

makedocs(
    sitename = "SimpleGrad.jl",
    modules = [SimpleGrad],
    format = Documenter.HTML(
        assets = ["assets/custom.css"]
    ),    pages = [
        "Welcome" => "index.md",
        "Usage" => "usage.md",
        "Under the Hood" => "under_the_hood.md",
        "Tutorials" => [
            "Linear Regression" => "tutorials/linear_regression.md",
            "MNIST" => "tutorials/mnist.md",
        ],
    ],
    sitename = "SimpleGrad.jl",
    root = "docs/build"
)


deploydocs(
    repo = "github.com/mikesaint-antoine/SimpleGrad.jl.git",
    branch = "main",
    target = "docs/build"
)