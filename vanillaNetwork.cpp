//
//  vanillaNetwork.cpp
//  VanillaNet
//
//  Created by Riley Jones on 4/8/21.
//

#include "vanillaNetwork.hpp"
#include "dataManager.hpp"
#include "Helper.hpp"

//----------------------------Network----------------------------------------------------


neuralNet::neuralNet() {
    
    numHidden = 0;
    numLayers = 0;
    
    hidden = NULL;
    network = NULL;
    
    data = NULL;
    pastCost = NULL;
    pastGradCost = NULL;
    avgGradCost = -1;
    averageCost = -1;
    batchSize = 1;
    batchCount = 0;
    index = 0;
    
    name = "untitled";
    
    forwardCount = 0;
    
}

neuralNet::~neuralNet() {
}

int neuralNet::configureNetwork() {
    
    input = new inputLayer(settings.inputSize);
    output = new outputLayer(settings.outputSize);
    
    //Settings
    name = settings.name;
    outputSize = settings.outputSize;
    inputSize = settings.inputSize;
    learningRate = settings.learningRate;
    batchSize = settings.batchSize;
    numLayers = settings.numOfLayers;

    if (numLayers <= 2) {
        std::cout << "Error: not enough layers in neural net" << std::endl;
    }
    numHidden = settings.numOfLayers - 2;
    if(numHidden > 0) {
        delete hidden;
        hidden = new wiredLayer*[numHidden];
        
        for(int i = 0; i < numHidden; i++) {
            hidden[i] = new wiredLayer(settings.layers[i]);
        }
    }
    
    delete network;
    network = new layer*[numLayers];
    network[0] = input;
    for (int i = 0; i < numHidden; i++) {
        network[1+i] = hidden[i];
    }
    network[numLayers-1] = output;

    
    return 1;
    
}

int neuralNet::connectData(dataManager &manager) {
    data = &manager;
    
    if (input->size != data->imageSize) {
        std::cout << "Input layer size does not match image size" << std::endl;
        return 0;
    }
    if (output->size != data->labelSize) {
        std::cout << "Output layer size does not match label size" << std::endl;
        return 0;
    }
    return 1;
}


int neuralNet::setBatchRecords(int batch_size) {
    pastCost = new double[batch_size];
    pastGradCost = new double[batch_size];
    
    for (int i = 0; i < batch_size; i++) {
        pastCost[i] = 0;
        pastGradCost[i] = 0;
    }

    return 1;
}


int neuralNet::wireNetwork() {

    input->wireDataManager(*data, *(hidden[0]));
    
    for (int i = 0; i < numHidden; i++) {
        hidden[i]->wireNodes(*network[i], *network[i+2]);
    }
    output->wireOutput(*hidden[numHidden-1], *data);
    return 1;
}

int neuralNet::expressNetwork() {
    
    if(checkConnections() == 0) {return 0;}
    for (int i = 0; i < numLayers; i++) {
        network[i]->printLayerInfo();
    }
    
    return 1;
}

bool neuralNet::checkConnections() {
    if (input == NULL) {
        std::cout << "Input layer not valid" << std::endl;
        return 0;
    }
    if (output == NULL) {
        std::cout << "Output layer not valid" << std::endl;
        return 0;
    }
    if (data == NULL) {
        std::cout << "Data manager not valid" << std::endl;
        return 0;
    }
    if (hidden == NULL) {
        std::cout << "Hidden layers not valid" << std::endl;
        return 0;
    }
    for (int i = 0; i < numLayers; i++) {
        if (network[i]->checkConnections() == 0) {
            std::cout << "Layer: " << i << " not valid" << std::endl;
            return 0;
        }
    }
    
    return 1;
}

void neuralNet::setLayerValues(double value) {
    for (int i = 0; i <= numLayers; i++) {
        network[i]->setValues(value);
    }
}

void neuralNet::randomiseValues(double max, double min) {
    for (int i = 0; i < numLayers; i++) {
        network[i]->randomiseValues(max, min);
    }
}

layer* neuralNet::getLayer(int i) {
    if (i < 0) {i = 0;}
    if (i >= numLayers) {i = numLayers - 1;}
    
    return network[i];
}

void neuralNet::runforward() {
    if (checkConnections() == true) {
            for (int i = 0; i < numLayers; i++)  {
                network[i]->calcActivation();
            }
            forwardCount++;
    }
}

void neuralNet::trainNetwork(int times) {
    if (checkConnections() == true) {
        
        data->mode = TRAINING;
        for (int i = 0; i < times; i++) {
            runforward();
            calcGradientDecent();
    
        if  (addCost(output->calcCost()) == true) {
            //std::cout << "Batch complete,average cost: " << averageCost << std::endl;
                applyGradient();
                resetGradients();
                batchCount++;
            }
            
            data->nextTrain();
        }
    }
}
    

int neuralNet::addCost(double cost) {
    
    pastCost[index++] = cost;
    if(index >= batchSize) {
        
        double sum = 0;
        for (int i = 0; i < batchSize; i++) {
            sum = sum + pastCost[i];
        }
        averageCost = sum/batchSize;
        
        index = 0;
        return 1;
    }
    return 0;
}

void neuralNet::resetGradients() {
    for (int i = 0; i < numHidden; i ++) {
        hidden[i]->setGradient(0);
    }
    output->setGradient(0);
}

void neuralNet::calcGradientDecent() {
        
    //Steps to calculate the gradient matrices
    //Step1 Calculate the activation gradient of the layer pior or the cost
    //Step2 pass to layer before
    //Step3 Calculate and add the bais gradient
    //Step4 Calculate and add  the weight gradient
    //Step5 call the same on next layer until the input
    
    for(int i = numLayers-1; i > 0; i--) {
        network[i]->calcActivGrad();
        network[i]->calcWeightGrad();
        network[i]->calcBiasGrad();
    }
}

void neuralNet::applyGradient() {
    for (int i = 0; i < numHidden; i++) {
        hidden[i]->applyGradient(learningRate);
    }
    output->applyGradient(learningRate);
}

float neuralNet::calcAccuracy(int times, double confidence, bool extenedResponse) {
    
    data->mode = TESTING;
    int matches = 0;
    int* result = new int[outputSize];
    int* label  = new int[outputSize];


    int* digit_count = new int[10];
    int* digit_correct = new int[10];
    float* digit_accuracy = new float[10];
    for(int i = 0; i < 10; i++) {
        digit_count[i] = 0;
        digit_correct[i] = 0;
        digit_accuracy[i] = 0;
    }
    
    for(int i = 0; i < times; i++) {
        
        runforward();
        double* out = output->activations;
        double* answer = data->getCurrent()->label;
        
        pickTop(out, result, outputSize);
        discreteRounding(answer, label, outputSize, confidence);
        
        int parts = 0;
        int ans = -1;
        for (int j = 0; j < outputSize; j++) {
            if(result[j] == label[j]) {
                //std::cout << "Match" << std::endl;
                parts++;
            }
            if (label[j] == 1) {
                ans = j;
            }
        }
        digit_count[ans]++;

        if (extenedResponse == true) {
            int j = 0;
            for (;j < outputSize; j++) if (result[j] == 1) break;
            std::cout << "\nPrediction: " << j << endl;
            printVector(result, outputSize);
            printImage(*data, data->getCurrent());
            std::cout << std::endl;
        }
        

        if(parts == outputSize) {
            matches++;
            digit_correct[ans]++;
        } 
        
        data->nextTest();
    }
    for (int i = 0; i < 10; i++) {
        digit_accuracy[i] = (float)digit_correct[i] / (float)digit_count[i];
    }

    if (extenedResponse == true) {
        for (int i = 0; i < 10; i++) {
            cout << "Digit " << i << " Accuracy: " << digit_accuracy[i] << endl;
        }
    } else {
        cout << "Digit Accuracy: ";
        printVector<float>(digit_accuracy, 10);
    }

    cout << "Overall Accuracy: " << (float)matches / (float)times << endl;
    
    delete[] result;
    delete[] label;
    delete[] digit_count;
    delete[] digit_correct;
    delete[] digit_accuracy;
    
    //std::cout << matches << "  " << times << "  " << (float)matches / (float)times << std::endl;
    return (float)matches / (float)times;
}

double* neuralNet::runExample(double* example, int size) {
    
    data->mode = TESTING;
    if( size != inputSize) {
        std::cout  << "Incorrect example size" << std::endl;
        return NULL;
    }
    
    copyVector(example, input->activations, size);
    
    if (checkConnections() == true) {
            for (int i = 1; i < numLayers; i++)  {
                network[i]->calcActivation();
            }
            forwardCount++;
    }
    
    return output->activations;
}

int* neuralNet::predictAnswer(double* example, int size, int* out, double confidence) {
    discreteRounding(runExample(example, size), out, outputSize, confidence);
    return out;
    
}

void neuralNet::info() {
    std::cout << "Name: " << name << std::endl;
    std::cout << "Inputs: " << inputSize << std::endl;
    std::cout << "# Layers: " << numLayers << std::endl;
    std::cout << "Outputs: " << outputSize << std::endl;
}


// ----------------------------- Network Reader --------------------------------------

bool networkReader::saveNetwork(neuralNet &net, string location) {
    
    ofstream serial(location ,ios::binary | ios::out | ios::trunc);
    
    if (!serial.is_open()) {
        cout << "File could not be opened" << endl;
        return false;
    }
    std::cout << "File opened" << std::endl;

    serial.write((char*)&net.name, sizeof(string));
    serial.write((char*)&net.inputSize, sizeof(int));
    serial.write((char*)&net.outputSize, sizeof(int));
    serial.write((char*)&net.numLayers, sizeof(int));
    serial.write((char*)&net.learningRate, sizeof(double));
    serial.write((char*)&net.batchSize, sizeof(int));
    for (int i = 0; i < net.numHidden; i++) {
        serial.write((char*)&(net.hidden[i]->size), sizeof(int));
    }
    
    
    //Hidden layers
    for (int i = 0; i < net.numHidden; i++) {
        wiredLayer* layer = net.hidden[i];
        
        for (int i = 0; i < layer->size; i++) {
            for (int j = 0; j < layer->priorSize; j++) {
                serial.write((char*)&layer->weights[i][j], sizeof(double));
           }
        }
        
        for (int i = 0; i < layer->size; i++) {
            serial.write((char*)&layer->bias[i], sizeof(double));
        }
    }
    //Output layer
    for (int i = 0; i < net.output->size; i++) {
        for (int j = 0; j < net.output->priorSize; j++) {
            serial.write((char*)&net.output->weights[i][j], sizeof(double));
       }
    }
    
    for (int i = 0; i < net.output->size; i++) {
        serial.write((char*)&net.output->bias[i], sizeof(double));
    }
    
    serial.close();
    return false;
}
bool networkReader::readNetwork(neuralNet* net, string location, dataManager* data) {
    
    ifstream serial(location ,ios::binary | ios::in);
    
    if (!serial.is_open()) {
        cout << "File could not be opened" << endl;
        return false;
    }
    std::cout << "File opened" << std::endl;
    
    string name = "";
    serial.read((char*)&name, sizeof(string));
    int inputSize = 0;
    serial.read((char*)&inputSize, sizeof(int));
    int outputSize = 0;
    serial.read((char*)&outputSize, sizeof(int));
    int numLayers = 0;
    serial.read((char*)&numLayers, sizeof(int));
    double learningRate = 0;
    serial.read((char*)&learningRate, sizeof(double));
    int batchSize = 0;
    serial.read((char*)&batchSize, sizeof(int));

    std::cout << "Name: " << name << std::endl;
    std::cout << "Inputs: " << inputSize << std::endl;
    std::cout << "# Layers: " << numLayers << std::endl;
    std::cout << "Outputs: " << outputSize << std::endl;
    std::cout << std::endl;

    int* layers = new int[numLayers - 2];
    for (int i = 0; i < numLayers - 2; i++) {
        serial.read((char*)&(layers[i]), sizeof(int));
    }

    net->settings = (neuralNet::configuration){
        .name = name,
        .inputSize = inputSize,
        .outputSize = outputSize,
        .numOfLayers = numLayers,
        .learningRate= learningRate,
        .batchSize = batchSize,
        .layers = layers
    };
    net->configureNetwork();
    net->connectData(*data);
    net->setBatchRecords(batchSize);
    net->wireNetwork();
    
    //Hidden layers
    for (int i = 0; i < net->numHidden; i++) {
        wiredLayer* layer = net->hidden[i];
        
        for (int i = 0; i < layer->size; i++) {
            for (int j = 0; j < layer->priorSize; j++) {
                serial.read((char*)&layer->weights[i][j], sizeof(double));
           }
        }
        
        for (int i = 0; i < layer->size; i++) {
            serial.read((char*)&layer->bias[i], sizeof(double));
        }
    }
    //Output layer
    for (int i = 0; i < net->output->size; i++) {
        for (int j = 0; j < net->output->priorSize; j++) {
            serial.read((char*)&net->output->weights[i][j], sizeof(double));
       }
    }
    
    for (int i = 0; i < net->output->size; i++) {
        serial.read((char*)&net->output->bias[i], sizeof(double));
    }
    
    
    serial.close();
    
    return true;
}
