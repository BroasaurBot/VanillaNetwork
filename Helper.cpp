//
//  Helper.cpp
//  VanillaNet
//
//  Created by Riley Jones on 17/1/22.
//

#include "Helper.hpp"

/// ------------------------------------------------------Helper functions --------------------------------------------------------------
//prints any 2D double matrix

//Prints any 3D matrix to the screen row and colum
void printMat(double** matrix, int n , int m) {
    std::cout << "[";
    for (int i = 0; i < n-1; i++) {
        std::cout << "[";
        for (int j = 0; j < m-1; j++) {
            std::cout << matrix[i][j] << ", ";
        }
        std::cout << matrix[i][m-1] << "]," << std::endl;
    }
    std::cout << "[";
    for (int j = 0; j < m-1; j++) {
        std::cout << matrix[n-1][j] << ", ";
    }
    std::cout << matrix[n-1][m-1] << "]]" << std::endl;
}


//Initialises the rand seed
void initRand(unsigned int seed) {
    if (seed == 0) {
        seed = static_cast<unsigned int>(time(0));
        srand(static_cast<unsigned int>(seed));
    }
    std::cout << "Seed: " << seed << std::endl;
    for(int x = 0; x < 20; x++) {rand();}
}

//gets a random number in range
double getRandomNumber(double min, double max) {
    static const double fraction = 1.0 / (RAND_MAX);
    return min + (max - min) * (fraction * rand());
}

//Copies the content of the 'copy' to the 'paste'
void copyVector(double* copy, double* paste, int amount) {
    for (int i = 0; i < amount; i++) {
        paste[i] = copy[i];
    }
}

//dot product of 2 2D matrices
bool dotMat(double** mat1, double** mat2, double** buffer, int n1, int m1, int n2, int m2) {
    
    if (m1 != n2) {
        std::cout << "m1 not match n2" << std::endl;
        return -1;
    }
    int n = n1;
    int m = m2;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double sum = 0;
            
            for (int p = 0; p < m1; p++) {
                sum = sum + mat1[i][p] * mat2[p][j];
            }
            buffer[i][j] = sum;
        }
    }
    
    return 1;
}

//dot product of a 2D and 1D matrix in that order
double elemDotMat(double** mat1, double* mat2, int n1, int m1, int n2, int m2, int i_i) {
    if (m1 != n2) {
        std::cout << "m1 not match n2" << std::endl;
        return -1;
    }
    
    int i = i_i;
    double sum = 0;
    for (int k = 0; k < m1; k++) {
        sum = sum + mat1[i][k] * mat2[k];
    }
    return sum;
}

bool dotMat(double** mat1, double* mat2, double* buffer, int n1, int m1, int n2, int m2) {
    if (m1 != n2) {
        std::cout << "m1 not match n2" << std::endl;
        return -1;
    }
    int n = n1;
    
    for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int p = 0; p < m1; p++) {
                sum = sum + mat1[i][p] * mat2[p];
            }
            buffer[i] = sum;
    }
    
    return 1;
}

//Adds the content of mat1 to mat2
bool addVector(double* mat1, double* mat2, int size) {
    for (int i = 0; i < size; i++) {
        mat2[i] = mat2[i] + mat1[i];
    }
    return 1;
}

bool addMat(double** mat1, double** mat2, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            mat2[i][j] = mat2[i][j] + mat1[i][j];
        }
    }
    return 1;
}


//A shortened version of the sigmoid function
//Squeezes the num to the -1 - 1
double fast_sig(double num) {
    return num / (1 + abs(num));
}
double inverse_sig(double num) {
    return 1/((1+abs(num))*(1+abs(num)));
}


//Applies a function to a entire 1D array
bool applyFunc(double* mat,double (*function)(double), int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = function(mat[i]);
    }
    return 1;
}


void discreteRounding(double* num, int* out, int size, double boundary) {
    for (int i = 0; i < size; i++) {
        out[i] = num[i] > boundary ? 1 : 0;
    }
}

void pickTop(double* num, int* out, int size, double boundary) {
    
    int index = 0;
    for (int i = 0; i < size; i++) {
        if (num[i] > num[index]) {
            index = i;
        }
    }
    for (int i = 0; i < size; i++) {
        if (i == index) {
            if (num[i] >= boundary) out[i] = 1;
            else out[i] = num[i];
        }else {
            out[i] = 0;
        }
    }
    
}
