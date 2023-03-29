# VanillaNetwork
  Simple C++ implementation of a neural network.
  The network includes:
  - configurable layer size, and number of layers
  - Accuracy and training functions
  - learning rate, batchsize adjustments
  - reads input from standard IDX file format
  - Read and Save existing neural network to binary

With 3 layers and including a hidden layer with the size of 20 as capable to achieve 90% accuracy on the MNIST handwritten digit database. Load this neural network with networkReader::readNetwork(&_neuralNet_, "SavedNetworks/90.bin", _dataManager_);

This project was built of the C++ Standard library, and requires no external packages. Theory behind network was devised from author '3Blue1Brown', link to his content: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

