#include <iostream>
#include "nn.h"
#include "matrix.h"

using namespace std;

int main() {
    NeuralNetwork network(2, 10, 1);

    vector<vector<float>> inputs = {
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {0.0f, 0.0f},
        {1.0f, 1.0f}
    };

    vector<vector<float>> targets = {
        {1.0f},
        {1.0f},
        {0.0f},
        {0.0f}
    };

    int epochs = 10000;
    vector<int> indices = {0, 1, 2, 3};

    for (int epoch = 0; epoch < epochs; epoch++) {
        random_shuffle(indices.begin(), indices.end());

        for (int idx : indices) {
            network.train(inputs[idx], targets[idx]);
        }
    }

    network.predict({1, 0});
    network.predict({0, 1});
    network.predict({1, 1});
    network.predict({0, 0});
}