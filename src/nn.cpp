#include <iostream>
#include "nn.h"
#include "matrix.h"

using namespace std;

// initialize neural network architecture
NeuralNetwork::NeuralNetwork(int numInputs, int numHidden, int numOutputs, float learningRate) {
    this->numInputs = numInputs;
    this->numHidden = numHidden;
    this->numOutputs = numOutputs;
    this->learningRate = learningRate;

    // initialize weights and biases for input->hidden layer
    weightsInputHidden = new Matrix(numHidden, numInputs);
    weightsInputHidden->randomize();
    biasHidden = new Matrix(numHidden, 1);
    biasHidden->randomize();

    // initialize weights and biases for hidden->output layer
    weightsHiddenOutput = new Matrix(numOutputs, numHidden);
    weightsHiddenOutput->randomize();
    biasOutput = new Matrix(numOutputs, 1);
    biasOutput->randomize();
}

// cleanup pointers
NeuralNetwork::~NeuralNetwork() {
    delete weightsInputHidden;
    delete biasHidden;
    delete weightsHiddenOutput;
    delete biasOutput;
}

// predict the output of a given input
vector<float> NeuralNetwork::predict(const vector<float>& inputArray) {
    // convert input vector to matrix
    Matrix input = Matrix(inputArray); 

    // generate the hidden outputs
    Matrix hidden = Matrix::multiply(*weightsInputHidden, input);
    hidden.add(*biasHidden);  // add bias
    hidden.map(sigmoid);  // apply activation function!

    // generate the output outputs
    Matrix output = Matrix::multiply(*weightsHiddenOutput, hidden);
    output.add(*biasOutput);  // add bias
    output.map(sigmoid);  // apply activation function!

    // convert the output matrix to a vector
    Matrix outputTranspose = Matrix::transpose(output);
    return outputTranspose.toArray()[0];
}

// adjust weights and biases to fit a given input and target
void NeuralNetwork::train(const vector<float>& inputArray, const vector<float>& targetArray) {
    // convert input & target vectors to matrices
    Matrix input = Matrix(inputArray);
    Matrix target = Matrix(targetArray);

    // generate the hidden outputs
    Matrix hidden = Matrix::multiply(*weightsInputHidden, input);
    hidden.add(*biasHidden);  // add bias
    hidden.map(sigmoid);  // apply activation function!

    // generate the output outputs
    Matrix output = Matrix::multiply(*weightsHiddenOutput, hidden);
    output.add(*biasOutput);  // add bias
    output.map(sigmoid);  // apply activation function!

    // calculate the output errors
    Matrix outputError = error(target, output);

    // calculate the output gradients
    Matrix outputGradient = Matrix::map(output, dsigmoid);
    outputGradient.multiply(outputError);
    outputGradient.multiply(learningRate);

    // calculate hidden->output deltas
    Matrix hiddenTranspose = Matrix::transpose(hidden);
    Matrix weightsHiddenOutputDelta = Matrix::multiply(outputGradient, hiddenTranspose);

    // adjust weights and biases for hidden->output layer
    weightsHiddenOutput->add(weightsHiddenOutputDelta);
    biasOutput->add(outputGradient); // the delta of the bias is just the output gradient

    // calculate the hidden errors
    Matrix weightsHiddenOutputTranspose = Matrix::transpose(*weightsHiddenOutput);
    Matrix hiddenError = Matrix::multiply(weightsHiddenOutputTranspose, outputError);

    // calculate the hidden gradients
    Matrix hiddenGradient = Matrix::map(hidden, dsigmoid);
    hiddenGradient.multiply(hiddenError);
    hiddenGradient.multiply(learningRate);

    // calculate input->hidden deltas
    Matrix inputTranspose = Matrix::transpose(input);
    Matrix weightsInputHiddenDelta = Matrix::multiply(hiddenGradient, inputTranspose);

    // adjust weights and biases for input->hidden layer
    weightsInputHidden->add(weightsInputHiddenDelta);
    biasHidden->add(hiddenGradient); // the delta of the bias is just the hidden gradient
}

// activation function
float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// derivative of activation function
float dsigmoid(float y) {
    return y * (1 - y);
}

// calculate error
Matrix error(Matrix& target, Matrix& output) {
    // error = target - output
    return Matrix::map(target, [&output](float value, int i, int j) { return value - output.get(i, j); });
}