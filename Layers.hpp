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

//Base layer class, all other layers inherit from this
class layer {
private:
    
public:
    int size;           //Number of nodes in the layer
    bool wired;        
    double* activations; //Array of activations
    
    layer* priorLayer;  
    layer* afterLayer;
    
    layer(int size);
    ~layer();
    
    bool checkConnections();
    
    void virtual printLayerInfo();
    void virtual setValues(double value);   //Sets all the activations to a value
    void virtual randomiseValues(double max, double min); //Randomises the activations
    void virtual calcActivation();        //Calculates the activations of the layer, according to the prior layer
    
    void virtual calcActivGrad();
    void virtual calcWeightGrad();
    void virtual calcBiasGrad();
    
};


//Input layer, recieves input from a dataManager
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
    double** weights; //2D array of weights
    int priorSize; //The size of the layer before
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
    
    //Backprogation functions
    void setGradient(double num);
    void virtual calcActivGrad();
    void virtual calcWeightGrad();
    void virtual calcBiasGrad();

    //Applies the gradient to the weights and biases, after a complete batch.
    //Scale is the rate of learning 
    void applyGradient(double scale);
};

//Output layer determines cost function by comparing activations to the label from dataManager
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
