# VanillaNetwork
## A C++ implementation of a multi-layer perceptron

### Outline:
The VanillaNetwork project is capable of reading IDX formatted binaries, constructing configurable size multi-layer percetptron, training and testing the network.

### Description:
[VanillaNetwork](https://github.com/BroasaurBot/VanillaNetwork/blob/main/vanillaNetwork.hpp), is comprised of atleast one input, hidden and output layer. The number and size of the hidden layers is configurable. Input layer activations are read from the [DataManger](https://github.com/BroasaurBot/VanillaNetwork/blob/main/dataManager.hpp), matrix multiplication propogates activations throughtout hidden layers, output layer is compared against label data in DataManager. 

[Layers](https://github.com/BroasaurBot/VanillaNetwork/blob/main/Layers.hpp) are comprised of three types, InputLayer, WiredLayer, OutputLayer. WiredLayer performs backwards propogation after every training example, after every batch the weights and bias of the network are updated to improve performance of the network.

[DataManager](https://github.com/BroasaurBot/VanillaNetwork/blob/main/dataManager.hpp) is setup to read from IDX binary files. Current training example is the [MNIST Database](https://en.wikipedia.org/wiki/MNIST_database). Training examples are loaded by **createDigitIDX**, passing either the training or the testing dataset.

With 1 hidden layer of size 20 a trained network is capable to achieve 90% accuracy on the MNIST handwritten digit database. Load this neural network with networkReader::readNetwork(&_neuralNet_, "SavedNetworks/90.bin", _dataManager_);

This project was built of the C++ Standard library, and requires no external packages. Theory behind network was devised from author '3Blue1Brown', link to his content: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

![General Description of a multi-layer perception neural network](https://www.dtreg.com/uploaded/pageimg/MLFNwithWeights.jpg)




