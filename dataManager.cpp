//
//  DataManager.cpp
//  VanillaNet
//
//  Created by Riley Jones on 11/1/22.
//

#include "dataManager.hpp"
#include "vanillaNetwork.hpp"

using namespace std;

int readBytes(ifstream &file, int bytes) {
    int32_t num;
    int32_t whole = 0;
    for(int i = 0; i < bytes; i++){
        num = 0;
        file.read((char*)&num, 1);
        whole <<= 8;
        whole |= num;
    }
    return whole;
}

example* createBlankExamples(int num, int imageSize, int labelSize) {
    example* examples = new example[num];
    for (int i = 0; i < num; i++) {
        examples[i].image = new double[imageSize];
        examples[i].label = new double[labelSize];
    }
    return examples;
}

void printImage(dataManager &data, example *image) {
   
    std::cout << "Image:" << std::endl;
    std::cout << "-------------------------" << std::endl;
    for (int j = 0; j < data.imageSize; j++) {
        
        if ( j % 28 == 0) std::cout << std::endl;
        if (image->image[j] == 0) {
            std::cout << " ";
        }else {
            std::cout << (int)(image->image[j] * 9);
        }
    }
    std::cout << "\n-------------------------" << std::endl;
    std::cout << "Label: ";
    printVector(image->label, data.labelSize);
}

//---------------------------------------DataManager----------------------------------

dataManager::dataManager() {
    
    imageSize = -1;
    labelSize = -1;
    
    testIndex = 0;
    trainIndex = 0;
    
    mode = Mode::TESTING;
}

dataManager::~dataManager() {
}



bool dataManager::createDigitIDX(example_set &set, string images_path, string labels_path, int count) {
    
    ifstream images(workingDirectory + images_path, std::ios::in | ios::binary);
    ifstream labels(workingDirectory + labels_path, std::ios::in | ios::binary);
    
    if (!images.is_open()) {
        cout << "Images could not be opened" << endl;
        return false;
    }
    if (!labels.is_open()) {
        cout << "Labels could not be opened " << endl;
        return false;
    }
    
    images.seekg(8, ios::beg);
    labels.seekg(8, ios::beg);
    
    //first 2 4 byte integers are the the size of the image
    imageSize = readBytes(images, 4) * readBytes(images, 4);
    labelSize = 10;
    set.num = count;
    
    set.e = createBlankExamples(count, imageSize, labelSize);
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < imageSize; j++) {
            set.e[i].image[j] = readBytes(images, 1)/255.0;
        }
        for (int j = 0; j < labelSize; j++) {
            set.e[i].label[j] = 0;
        }
        set.e[i].label[readBytes(labels, 1)] = 1;
    }
    set.current = &set.e[0];
    
    return true;
}

bool dataManager::nextTest() {
    testIndex++;
    if (testIndex >= testingEx.num) {
        testIndex = 0;
        testingEx.current = &testingEx.e[testIndex];
        return true;
    }
    testingEx.current = &testingEx.e[testIndex];
    return false;
}

bool dataManager::nextTrain() {
    trainIndex++;
    if (trainIndex >= trainingEx.num) {
        trainIndex = 0;
        trainingEx.current = &trainingEx.e[trainIndex];
        return true;
    }
    trainingEx.current = &trainingEx.e[trainIndex];
    return false;
}

example* dataManager::getCurrentTest() {
    return testingEx.current;
}

example* dataManager::getCurrentTrain() {
    return trainingEx.current;
}

example* dataManager::getCurrent() {
    if (mode == Mode::TRAINING) {
        return getCurrentTrain();
    }else {
        return getCurrentTest();
    }
}
