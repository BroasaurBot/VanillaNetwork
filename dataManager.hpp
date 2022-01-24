//
//  DataManager.hpp
//  VanillaNet
//
//  Created by Riley Jones on 11/1/22.
//

#ifndef dataManager_hpp
#define dataManager_hpp

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <istream>
#include <iostream>
#include <bitset>

using namespace std;

const string workingDirectory = "/Users/rileyjones/Documents/Coding/C++/VanillaNet/VanillaNet/";

struct example {
    double* image;
    double* label;
};

struct example_set {
    example* e = NULL;
    example* current = NULL;
    int num = 0;
};

enum Mode {
    TRAINING,
    TESTING
};

int readBytes(ifstream& file, int bytes);
example* createBlankExamples(int num, int iamgeSize, int labelSize);

//----------------Data Manager --------------------------------------

class dataManager {
private:
    int numExamples;
    
public:
    
    example_set testingEx;
    example_set trainingEx;
    
    int imageSize;
    int labelSize;
    
    int testIndex;
    int trainIndex;
    
    dataManager();
    ~dataManager();
    
    bool createDigitIDX( example_set &set, string image, string labels, int count);
    
    bool nextTest();
    bool nextTrain();
    
    example* getCurrentTest();
    example* getCurrentTrain();
    example* getCurrent();
    
    Mode mode;
};

void printImage(dataManager &data, example *image);

#endif /* DataManager_hpp */
