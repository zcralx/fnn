cmake_minimum_required(VERSION 3.26)
project(Network_Plus)

set(CMAKE_CXX_STANDARD 23)

add_executable(Network_Plus main.cpp
        Neuron.cpp
        Neuron.h
        Utils.cpp
        Utils.h
        Layer.cpp
        Layer.h
        Test.cpp
        Test.h
        FeedforwardNeuralNetwork.cpp
        FeedforwardNeuralNetwork.h)
