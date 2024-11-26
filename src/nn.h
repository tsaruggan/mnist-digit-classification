#ifndef NN_H
#define NN_H

#include "matrix.h"
using namespace std;

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
        NeuralNetwork(int numInputs, int numHidden, int numOutputs);
        ~NeuralNetwork();

        vector<float> predict(const vector<float>& inputArray);
        void train(const vector<float>& inputArray, const vector<float>& targetArray, float learningRate = 0.1);

        vector<char> serialize() const;
        NeuralNetwork(const vector<char>& serializedData);
};

static float sigmoid(float x);
static float dsigmoid(float y);
static Matrix error(Matrix& target, Matrix& output);

#endif // NN_H
