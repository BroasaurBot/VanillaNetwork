//
//  vanillaNetwork.hpp
//  VanillaNet
//
//  Created by Riley Jones on 4/8/21.
//

#ifndef vanillaNetwork_hpp
#define vanillaNetwork_hpp

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include "dataManager.hpp"
#include "Layers.hpp"
#include "Helper.hpp"
#include <fstream>


//---------------------------Network--------------------------------------
class neuralNet {
private:
    
    int numHidden;
    int numLayers;
    int inputSize;
    int outputSize;
    
    
    double averageCost;
    double* pastCost;
    int batchCount;
    int index;
    
    double avgGradCost;
    double* pastGradCost;

    int forwardCount;
    
    string name;
    
    //Layers of the network
    inputLayer* input;
    wiredLayer** hidden;
    outputLayer* output;
    layer** network; //All the layers combined
    
    dataManager* data;
    
    int addCost(double cost);
    void calcGradientDecent();
    
    
public:
    
    struct configuration {
        string name;
        int inputSize;
        int outputSize;
        int numOfLayers;
        double learningRate;//rate of learning
        int batchSize;
        int* layers;
    };
    
    int batchSize;
    double learningRate;
    
    // Neural net configuration
    neuralNet();
    ~neuralNet();
    int configureNetwork();
    int connectData(dataManager &manager);
    int setBatchRecords(int batch_size);
    int wireNetwork();
    
    //Regular operations
    int expressNetwork();
    bool checkConnections();
    void setLayerValues(double valued);
    void randomiseValues(double max, double min);
    layer* getLayer(int i);
    wiredLayer* getHidden(int i);
    
    //Basic network trinaing
    void runforward();
    void trainNetwork(int times = 1);
    void resetGradients();
    void applyGradient();
    
    //Testing functions
    float calcAccuracy(int times, double confidence, bool displayFailed = false);
    double* runExample(double* example, int size);
    int* predictAnswer(double* example, int size, int* out, double confidence);

    void info();
    
    struct configuration settings;
    friend class networkReader;
};


// --------------------- File Reader -----------------------------------

class networkReader {
public:
    static bool saveNetwork(neuralNet &net, string location);
    static bool readNetwork(neuralNet* net, string location, dataManager* data);
};

#endif /* vanillaNetwork_hpp */
