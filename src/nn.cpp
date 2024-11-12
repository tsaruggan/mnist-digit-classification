#include <iostream>
#include "matrix.cpp"

using namespace std;

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

class NeuralNetwork {
    private:
        int numInputs;
        int numHidden;
        int numOutputs;

        Matrix* weightsInputHidden;
        Matrix* biasHidden;

        Matrix* weightsHiddenOutput;
        Matrix* biasOutput;

    public:
        NeuralNetwork(int numInputs, int numHidden, int numOutputs) {
            this->numInputs = numInputs;
            this->numHidden = numHidden;
            this->numOutputs = numOutputs;

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

        ~NeuralNetwork() {
            delete weightsInputHidden;
            delete biasHidden;
            delete weightsHiddenOutput;
            delete biasOutput;
        }

        vector<vector<float>> feedForward(vector<float>& inputArray) {
            // convert input vector to matrix
            Matrix input(inputArray); 

            // generate the hidden outputs
            Matrix hidden = Matrix::multiply(*weightsInputHidden, input);
            hidden.add(*biasHidden);  // add bias
            hidden.map(sigmoid);  // apply activation function

            // generate the output outputs
            Matrix output = Matrix::multiply(*weightsHiddenOutput, hidden);
            output.add(*biasOutput);  // add bias
            output.map(sigmoid);  // apply activation function

            // convert the output matrix to a vector
            return output.toArray();
        }
};

int main() {
    NeuralNetwork network = NeuralNetwork(2, 2, 1);
    vector<float> input = {1.0f, 0.0f};
    vector<vector<float>> output = network.feedForward(input);

    for (int i = 0; i < output.size(); i++) {
        for (int j = 0; j < output[i].size(); j++) {
            cout << output[i][j] << " ";
        }
        cout << endl;
    }
}