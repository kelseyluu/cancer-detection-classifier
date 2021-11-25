# cancer-detection-classifier
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JxAcqfiqNucH36TK1eat7cisJTItAhR0)
![GitHub last commit](https://img.shields.io/github/last-commit/kelseyluu/cancer-detection-classifier)

## About
Convolutional-recurrent neural network for detecting cancer from cell-free DNA whole genome sequencing data. 

### Inputs and Outputs
  1. Input is a 1kb resolution copy number (CN) profile and windowed protection score (WPS) profile for a 1Mb genomic region.
      * shape `(?, 1, 1000, 2)`
  2. Labels are binary indicators of the classes [1,0] for [cancer, no cancer].
      * shape `(?, 1)`
  3. Output is the sigmoid probability for the cancer class.
      * shape `(?, 1)`

### Model Architecture
  1. Convolutional layer with
      * 320 kernels
      * kernel length: 26
      * step size: 1
      * ReLU activation
      * padding: `same`
  2. Max-pooling layer with
      * pooling size: 13
      * pooling stride: 13
      * padding: `same`
  3. Dropout Layer with
      * rate: 0.2
  4. Bi-directional long short term memory layer
      * 320 forward neurons
      * 320 backward neurons
  5. Dropout Layer with
      * rate: 0.5
  6. Fully connected layer with
      * 925 neurons
      * ReLU activation
  7. Fully connected layer with
      * 1 neuron (output)
      * sigmoid activation

Objective Function: binary cross entropy

## License
This project is licensed under the terms of the MIT license.
