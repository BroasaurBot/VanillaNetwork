//
//  Layers.hpp
//  VanillaNet
//
//  Created by Riley Jones on 17/1/22.
//

#ifndef Layers_hpp
#define Layers_hpp

#include <stdio.h>
#include <iostream>
#include "dataManager.hpp"
#include "Helper.hpp"

//-----------------LAYERS-------------------------------------------------

class layer {
private:
    
public:
    int size;
    bool wired;
    double* activations;
    
    layer* priorLayer;
    layer* afterLayer;
    
    layer(int size);
    ~layer();
    
    bool checkConnections();
    
    void virtual printLayerInfo();
    void virtual setValues(double value);
    void virtual randomiseValues(double max, double min);
    void virtual calcActivation();
    
    void virtual calcActivGrad();
    void virtual calcWeightGrad();
    void virtual calcBiasGrad();
    
};



class inputLayer : public layer {
protected:
    
    
public:
    inputLayer(int size);
    ~inputLayer();
    dataManager* inputData;
    
    int wireDataManager(dataManager &inputData, layer &afterLayer);
    void virtual printLayerInfo();
    void virtual calcActivation();
};


class wiredLayer : public layer {
private:
    void giveAGrad(double* grad);
    
public:
    double** weights;
    int priorSize;
    double* bias;
    
    double** weightGrad;
    double* biasGrad;
    double* activationGrad;
    
    
    wiredLayer(int size);
    ~wiredLayer();
    
    int wireNodes(layer &prior, layer &after);
    void virtual printLayerInfo();
    void virtual setValues(double value);
    void virtual randomiseValues(double max, double min);
    void printInfo(bool weight = 0, bool bias = 0, bool wGrad = 0, bool bGrad = 0, bool aGrad = 0);
    void virtual calcActivation();
    
    void setGradient(double num);
    void virtual calcActivGrad();
    void virtual calcWeightGrad();
    void virtual calcBiasGrad();
    void applyGradient(double scale);
};

class outputLayer : public wiredLayer {
private:
    
public:
    
    dataManager *outputData;
    
    outputLayer(int size);
    ~outputLayer();
    
    
    int wireOutput(layer &prior, dataManager &output);
    void virtual printLayerInfo();
    double calcCost();
    void calcGradCost();
    void virtual calcActivGrad();
};


#endif /* Layers_hpp */
