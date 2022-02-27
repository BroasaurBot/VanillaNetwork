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

//Name of the neural network to open
//If the network does not exist inside the SavedNetworks folder a new network will be created
const string name = "Optimal";
const int total = 60000;
const int trains = 200;

int main(int argc, const char * argv[]) {
    
    
    //initialise the rand function
    initRand();

    dataManager data = dataManager();
    //Read in the MNST IDX file format
    data.createDigitIDX( data.trainingEx, "MNIST Database/train-images-idx3-ubyte", "MNIST Database/train-labels-idx1-ubyte",59000);
    data.createDigitIDX( data.testingEx, "MNIST Database/train-images-idx3-ubyte", "MNIST Database/train-labels-idx1-ubyte",40000);
    
    neuralNet nnet = neuralNet();
    if (!networkReader::readNetwork(&nnet, "SavedNetworks/" + name + ".bin", data)) {
        
        //Settings for a new neural network
        nnet.settings = {
            .name = "Handwritten Network",
            .inputSize = data.imageSize,
            .outputSize = data.labelSize,
            .numOfLayers = 3,               //refers to the number of layers including the input and ouput layers, must > 2
            .learningRate= -0.02,           // Generally values less than 0.5 work efficiently, must be negative
            .batchSize = 50,                //Number of tests run before performing backpropogation
            .layers = (int[]){32}
        };
        
        nnet.configureNetwork();
        nnet.connectData(data);
        nnet.wireNetwork();
        nnet.checkConnections();
        nnet.randomiseValues(-1, 1);
        
    } else {
        //If the network exists
    }
    
    
    cout << nnet.calcAccuracy(40000, 0.8, false) << endl; // Display the initial accuracy
    
     for (int i = 0; i < (total / trains); i++) {
        cout << "Examples: " << i * trains << endl;
        cout << "Accuracy: " << nnet.calcAccuracy(800, 0.8) << endl;
        nnet.trainNetwork(trains);
         
         if (i % 15 == 14) networkReader::saveNetwork(nnet, "SavedNetworks/" + name + ".bin"); //Saves the network after 15 sets of trains
    }
    
    //Final Accuracy Test
    data.testIndex =  0;
    cout << "New network: " << nnet.calcAccuracy(40000, 0.8, true) << endl;
    networkReader::saveNetwork(nnet, "SavedNetworks/" + name + "Final.bin"); //Saves the final of the network
    
    return 1;
}


