//
//  Layers.cpp
//  VanillaNet
//
//  Created by Riley Jones on 17/1/22.
//

#include "Layers.hpp"
#include "vanillaNetwork.hpp"
#include "Helper.hpp"

//-----------------------LAYERS --------------------------------------------------

layer::layer(int size) {
    this->size = size;
    activations = new double[size];
    wired = 0;
    
}

layer::~layer() {
    delete activations;
}

//Prints info of the layer
void layer::printLayerInfo() {
    std::cout << "Base layer: " << std::endl;
    std::cout << "activations size: "<< size << std::endl;
    printVector(activations, size);
    std::cout << "End layer:\n" << std::endl;
    
    if( afterLayer == NULL) {
        std::cout << "End of network." << std::endl;
    }
    
}

bool layer::checkConnections() {
    if (activations == NULL) {
        std::cout << "Activation nodes not valid" << std::endl;
        return 0;
    }
    if (wired == 0) {
        std::cout << "layer not wired" << std::endl;
        return 0;
    }
    return 1;
}

void layer::setValues(double value) {
    for(int i = 0; i < size; i++) {
        activations[i] = value;
    }
}

//randomises the value of the activations array
void layer::randomiseValues(double max, double min) {
    this->setValues(0);
}

void layer::calcActivation() {
    std::cout << "Calling incorrect layer activation calc" << std::endl;
}

void layer::calcActivGrad() {
    std::cout << "Calling incorrect layer activation gradient calc" << std::endl;
}
void layer::calcWeightGrad() {
    std::cout << "Calling incorrect layer weight gradient calc" << std::endl;
}
void layer::calcBiasGrad() {
    std::cout << "Calling incorrect layer bias gradient calc" << std::endl;
}


inputLayer::inputLayer(int size) : layer(size) {
    inputData = NULL;
    afterLayer = NULL;
}

inputLayer::~inputLayer() {}

int inputLayer::wireDataManager(dataManager &inputData, layer &afterLayer) {
    this->inputData = &inputData;

    if (inputData.imageSize != size) {
        std::cout << "The image does not match the size of the input layer" << std::endl;
        return -1;
    }
    
    this->afterLayer = &afterLayer;
    wired = 1;
    return 1;
}

void inputLayer::printLayerInfo() {
    std::cout << "Input layer: " << std::endl;
    layer::printLayerInfo();
    
}

void inputLayer::calcActivation() {
    double* image = inputData->getCurrent()->image;
    copyVector(image, activations, size);
}

//Wired layer
wiredLayer::wiredLayer(int size) : layer(size) {
    weights = NULL;
    priorSize = -1;
    bias = NULL;
    
    weightGrad = NULL;
    biasGrad = NULL;
    activationGrad = NULL;
    
    priorLayer = NULL;
    afterLayer = NULL;
}

wiredLayer::~wiredLayer() {
    delete weights;
    delete bias;
    
    delete weightGrad;
    delete biasGrad;
    delete activationGrad;
}

int wiredLayer::wireNodes(layer &prior, layer &after) {
    
    priorLayer = &prior;
    afterLayer = &after;
    
    activationGrad = new double[size];
    priorSize = priorLayer->size;

    weights = new double*[size];
    for(int i = 0; i < size; ++i)
        weights[i] = new double[prior.size];
    weightGrad = new double*[size];
    for(int i = 0; i < size; ++i)
        weightGrad[i] = new double[prior.size];
    
    bias = new double[size];
    biasGrad = new double[size];
    
    wired = 1;
    
    return 1;
}

void wiredLayer::printLayerInfo() {
    std::cout << "WiredLayer: " <<std::endl;
    printInfo(true, true,true,true,true);
    layer::printLayerInfo();
}

void wiredLayer::setValues(double value) {
    layer::setValues(value);
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < priorSize; j++) {
                weights[i][j] = value;
                weightGrad[i][j] = value;
        }
    }
    
    
    for (int i = 0; i < size; i++) {
        bias[i] = value;
        biasGrad[i] = value;
        activationGrad[i] = value;
        }
    }

void wiredLayer::randomiseValues(double max, double min) {
    layer::randomiseValues(max, min);
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < priorSize; j++) {
                weights[i][j] = getRandomNumber(min, max);
                weightGrad[i][j] = getRandomNumber(min, max);
        }
    }
    
    
    for (int i = 0; i < size; i++) {
        bias[i] = getRandomNumber(min, max);
        biasGrad[i] = getRandomNumber(min, max);
        activationGrad[i] = getRandomNumber(min, max);
    }
}

wiredLayer* neuralNet::getHidden(int i) {
    if (i < 0) {i = 0;}
    if (i >= numHidden) {i = numHidden - 1;}
    
    return hidden[i];

}

void wiredLayer::printInfo(bool weight, bool bias, bool wGrad, bool bGrad, bool aGrad) {
    if(weight ==  true) {
        std::cout << "Weights:" << std::endl;
        printMat(weights, size, priorSize);
    }
    if(bias == true) {
        std::cout << "Bias:" << std::endl;
        printVector(this->bias, size);
    }
    if(wGrad == true) {
        std::cout << "Weight-Gradient:" << std::endl;
        printMat(this->weightGrad, size, priorSize);
    }
    if(bGrad == true) {
        std::cout << "Bias-Gradient:" << std::endl;
        printVector(this->biasGrad, size);
    }
    if(aGrad == true) {
        std::cout << "Activation-Gradient:" << std::endl;
        printVector(this->activationGrad, size);
    }
}

void wiredLayer::calcActivation() {
    dotMat(weights, priorLayer->activations, activations, size, priorSize, priorSize, 1);
    addVector(bias, activations, size);
    applyFunc(activations, &fast_sig, size);
}

void wiredLayer::giveAGrad(double* grad) {
    //iterate through prior layer activations gradient matrix
    for(int k = 0; k < priorSize; k++) {
        
        //then for each of its connections to current layer
        double sum = 0;
        for (int j = 0; j < size; j++) {
            
            double z = elemDotMat(weights, priorLayer->activations, size, priorSize, priorSize, 1, j) + bias[j];
            sum = sum + (weights[j][k] * inverse_sig(z) * activationGrad[j]);
        }
        grad[k] = sum;
        
    }
}

void wiredLayer::calcActivGrad() {
    wiredLayer* layer_ptr = static_cast<wiredLayer*>(afterLayer);
    layer_ptr->giveAGrad(activationGrad);
}
void wiredLayer::calcWeightGrad() {
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < priorSize; k++) {
            double z = elemDotMat(weights, priorLayer->activations, size, priorSize, priorSize, 1, i) + bias[i];
            weightGrad[i][k] = weightGrad[i][k] + (priorLayer->activations[k] * inverse_sig(z) * activationGrad[i]);
        }
    }
    
}
void wiredLayer::calcBiasGrad() {
    
    for (int i = 0; i < size; i++) {
            double z = elemDotMat(weights, priorLayer->activations, size, priorSize, priorSize, 1, i) + bias[i];
            biasGrad[i] = biasGrad[i] + ( inverse_sig(z) * activationGrad[i]);
    }
}

void wiredLayer::setGradient(double num) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < priorSize; j++) {
            weightGrad[i][j] = num;
        }
        
        biasGrad[i] = num;
        activationGrad[i] = num;
    }
}

void wiredLayer::applyGradient(double scale) {
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < priorSize; j++) {
            weightGrad[i][j] = weightGrad[i][j] * scale;
        }
        
        biasGrad[i] = biasGrad[i] * scale;
    }
    
    addVector(biasGrad, bias, size);
    //printVector(bias, size);
    addMat(weightGrad, weights, size, priorSize);
    //printMat(weights, size, priorSize);
}


outputLayer::outputLayer(int size) : wiredLayer(size) {
    outputData = NULL;
}

outputLayer::~outputLayer() {}


int outputLayer::wireOutput(layer& prior, dataManager &output) {
    priorLayer = &prior;
    outputData = &output;
    
    activationGrad = new double[size];
    priorSize = prior.size;

    weights = new double*[size];
    for(int i = 0; i < size; ++i)
        weights[i] = new double[prior.size];
    weightGrad = new double*[size];
    for(int i = 0; i < size; ++i)
        weightGrad[i] = new double[prior.size];
    
    bias = new double[size];
    biasGrad = new double[size];
    
    wired = 1;
    return 1;
}

void outputLayer::printLayerInfo() {
    std::cout << "OutputLayer: " << std::endl;
    wiredLayer::printLayerInfo();
    
}

double outputLayer::calcCost() {
    double sum = 0;
    double* label = outputData->getCurrent()->label;
    
    for (int i = 0; i < size; i++) {
        sum = sum + (activations[i] - label[i]) * (activations[i] - label[i]);
    }
    
    return sum;
}
void outputLayer::calcActivGrad() {
    calcGradCost();
}

void outputLayer::calcGradCost() {
    //2(a - y)
    for(int i = 0; i < size; i++) {
        activationGrad[i] = 2 * (activations[i] - outputData->getCurrent()->label[i]);
        
        //std::cout << "Comparing: " << activations[i] << " and: " << outputData->getCurrent()->label[i] <<std::endl;
    }
    
    //std::cout << "Output calc grad: " << std::endl;
    //print2DMatrix(activationGrad, size);
}


    
