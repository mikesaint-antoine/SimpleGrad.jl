# MyMachineLearningPackage.jl

Welcome to the documentation for `MyMachineLearningPackage.jl`.



## Introduction

Hi! Thanks for checking out this Julia library!

This library is based on Andrej Karpathy's Micrograd (which is in Python), and has a similar goal: to provide basic gradient-tracking, with code **simple enough to read** and **easy enough to edit and customize.** 

Like the original Python Micrograd, we'll be using a *Value* class for scalar-based gradient tracking. 

In addition, I've also added a *Tensor* class for array/matrix-based gradient tracking. Even though this wasn't included in the original Micrograd, I decided to add it so we could do simple examples like MNIST without the code taking forever.

I'm hoping that this will be a useful resource for people who are just getting started with Julia and trying to learn the basics. 


## Installation

To install the package, you need to...

```julia
using Pkg
Pkg.add("TestPackage")
```


Right now there's no fancy installation with a package manager or anything. You need to actually download the *mikerograd.jl* file form this repo, put it somewhere on your computer, and then direct Julia to it's location in order to use it. Like this, but replace the location with its location on your computer:

```julia
push!(LOAD_PATH, "/Users/mikesaint-antoine/Desktop/new_mikerograd") 
# change this to the location of the folder where mikerograd.jl is on your computer

using mikerograd
```

## Very Basics
There are two types of objects in mikerograd: *Values* and *Tensors*.

