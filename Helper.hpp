//
//  Helper.hpp
//  VanillaNet
//
//  Created by Riley Jones on 17/1/22.
//

#ifndef Helper_hpp
#define Helper_hpp

#include <stdio.h>
#include <cstdlib>
#include <iostream>

//----------------General structs and functions-----------------------

template <typename T> void printVector(T* matrix, int size);
double getRandomNumber(double min, double max);
void initRand(unsigned int seed = 0);
void printMat(double** matrix, int n , int m);
bool dotMat(double** mat1, double** mat2, double** buffer, int n1, int m1, int n2, int m2);
bool dotMat(double** mat1, double** mat2, double* buffer, int n1, int m1, int n2, int m2);
bool dotMat(double** mat1, double* mat2, double* buffer, int n1, int m1, int n2, int m2);
bool addVector(double* mat1, double* mat2, int size);
void copyVector(double* copy, double* paste, int amount);
bool applyFunc(double* mat,double (*function)(double), int size);
double inverse_sig(double num);
double elemDotMat(double** mat1, double* mat2, int n1, int m1, int n2, int m2, int i_i);
bool addMat(double** mat1, double** mat2, int n, int m);
int discreteRounding(double num, double boundary);
void discreteRounding(double* in,int* out,int size ,double boundary);
void pickTop(double* num, int* out, int size ,double boundary = -1);
double fast_sig(double num);


template <typename T>
void printVector(T* vector, int size) {
    
    std::cout << "[";
    for (int x = 0; x < size-1; x++) {
        std::cout << vector[x] << ", ";
    }
    std::cout << vector[size-1] << "]" << std::endl;
}

#endif /* Helper_hpp */
