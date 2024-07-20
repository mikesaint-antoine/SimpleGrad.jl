# SimpleGrad.jl

[[Documentation Site]](https://mikesaint-antoine.github.io/SimpleGrad.jl)

[[Video Tutorials]](https://www.youtube.com/playlist?list=PLWVKUEZ25V97tNULapu07DhWv6_W4NfpE)

Hi, thanks for checking out `SimpleGrad.jl`! This is a gradient-tracking tool for basic machine learning applications, including neural nets. But unlike other ML packages, the primary goal of SimpleGrad is to be *educational.* The idea is that the source code should be easy enough to read, understand, and edit/customize, and the package should be both usable for basic applications and also a helpful for people who are learning Julia or ML (or both).

The [documentation site](https://mikesaint-antoine.github.io/SimpleGrad.jl) also includes an [Under the Hood](https://mikesaint-antoine.github.io/SimpleGrad.jl/under_the_hood/) section that explains how everything works, so that you can recreate it from scratch if you want to. My goal here was to write it like a textbook chapter, meant for people who like to understand how things work from first principles.

For people who prefer to learn from videos, I've also made a [Neural Nets from Scratch in Julia](https://www.youtube.com/playlist?list=PLWVKUEZ25V97tNULapu07DhWv6_W4NfpE) Youtube series explaining how to recreate this package from scratch and how everything works along the way.

By the way, this project is an ongoing work-in-progress and I'm open to suggestions, criticisms, questions, and pull-requests. You can reach me by email at *mikest@udel.edu*.

## Installation

If everything's working properly, you should be able to install SimpleGrad right from the Julia package manager like this:

```julia
using Pkg
Pkg.add("SimpleGrad")
```
As an alternative, you can also just grab the [source code file](https://github.com/mikesaint-antoine/SimpleGrad.jl/blob/main/src/SimpleGrad.jl) and put it on your computer. Then, just direct Julia to its location in order to use it. Like this, but replace the location with its location on your computer:

```julia
push!(LOAD_PATH, "/Users/mikesaint-antoine/Desktop/") 
# change this to the location of the folder where SimpleGrad.jl is on your computer

using SimpleGrad
```
In fact, even if installation through ```Pkg``` is working, grabbing the source code directly might be the better approach for educational purposes, so that you can easily edit it and play around with it.


## Credits

Huge thanks to these people for teaching me how to do this stuff:

- Andrej Karpathy for his [Micrograd tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0)
- Harrison Kinsley (Sentdex on Youtube) for his [Neural Net From Scratch in Python textbook](https://nnfs.io/)
- u/Bob_Dieter for his [reddit comments](https://www.reddit.com/r/Julia/comments/18knzll/comment/kdytys3)
