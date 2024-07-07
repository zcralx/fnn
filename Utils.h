//
// Created by rui zhan on 2024/5/28.
//

#ifndef NETWORK_PLUS_UTILS_H
#define NETWORK_PLUS_UTILS_H


#include "iostream"
#include "vector"

class Utils {
public:
    static double random_generator(double mu = 0.0);
    static void print(double);
    static void printVector(const std::vector<double>&);
    static void printVector(const std::vector<std::vector<double>>&);
    static double Sigmod(double x);
    static double meanSquaredError(const std::vector<double> predictions, const std::vector<double> true_outputs);
    static double binaryCrossEntropy(const std::vector<double> predictions, const std::vector<double> targets);
    static double addBy2Vector(const std::vector<double>&, const std::vector<double>&);
};

#endif //NETWORK_PLUS_UTILS_H
