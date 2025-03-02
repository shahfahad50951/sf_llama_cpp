# LLaMA Implementation from Scratch (C++)

## Goals
The goal of this project is to implement LLaMA model from META in C++ from scratch.
This includes building a tensor library from scratch to be used as the building block.
I undertook this project to understand more about the implementation of LLMs which will help me understand the nitty-gritty detail of LLMs and their underlying architecture.

## Inspiration
This project is heavily inspired by the [llama.cpp](https://github.com/ggml-org/llama.cpp) project however no portion of this code is copied.
All implementations are entirely my own thoughts. Expect a lot of refactoring because the code will be changed as and when required.

## Project Structure
1. Currently there is only one monolithic file `lib.cpp` that contains all the implementations and APIs. This will be broken into modules in future as and when I see fit.

## Compilation
All the APIs defined in `lib.cpp` can be accessed using another cpp file.
This project is using `main.cpp` to chain together APIs present in `lib.cpp` to get something useful.

To compile the `main.cpp` file, use the following command line on terminal
```
g++ -o <output_program_name> main.cpp
```
