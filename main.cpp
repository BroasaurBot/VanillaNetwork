 //
//  main.cpp
//  VanillaNet
//
//  Created by Riley Jones on 4/8/21.
//

#include <iostream>
#include <ctime>
#include <string>
#include <cstdlib>
#include "vanillaNetwork.hpp"
#include "dataManager.hpp"

using namespace std;

const string name = "Optimal";
const int total = 60000;
const int trains = 200;

int main(int argc, const char * argv[]) {
    
    initRand();
    dataManager data = dataManager();
    data.createDigitIDX( data.trainingEx, "MNIST Database/train-images-idx3-ubyte", "MNIST Database/train-labels-idx1-ubyte",59000);
    data.createDigitIDX( data.testingEx, "MNIST Database/train-images-idx3-ubyte", "MNIST Database/train-labels-idx1-ubyte",40000);
    
    neuralNet nnet = neuralNet();
    
    if (!networkReader::readNetwork(&nnet, "SavedNetworks/" + name + ".bin", data)) {
        
        nnet.settings = {
            .name = "Handwritten Network",
            .inputSize = data.imageSize,
            .outputSize = data.labelSize,
            .numOfLayers = 3,
            .learningRate= -0.02,
            .batchSize = 50,
            .layers = (int[]){32}
        };
        
        nnet.configureNetwork();
        nnet.connectData(data);
        nnet.wireNetwork();
        nnet.checkConnections();
        nnet.randomiseValues(-1, 1);
        
    } else {
        nnet.learningRate = -0.01;
    }
    
    cout << nnet.calcAccuracy(40000, 0.8, false) << endl;
    
     for (int i = 0; i < (total / trains); i++) {
        cout << "Examples: " << i * trains << endl;
        cout << "Accuracy: " << nnet.calcAccuracy(800, 0.8) << endl;
        nnet.trainNetwork(trains);
         
         if (i % 15 == 14) networkReader::saveNetwork(nnet, "SavedNetworks/" + name + ".bin");
    }
    
    data.testIndex =  0;
    cout << "New network: " << nnet.calcAccuracy(40000, 0.8, true) << endl;
    networkReader::saveNetwork(nnet, "SavedNetworks/" + name + "Final.bin");
    
    return 1;
}


