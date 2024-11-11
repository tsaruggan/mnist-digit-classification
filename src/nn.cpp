#include <iostream>
#include "matrix.cpp"

using namespace std;

class NeuralNetwork {
    int numInputs;
    int numHidden;
    int numOutputs;

    public: 
        NeuralNetwork(int numInputs, int numHidden, int numOutputs) {
            this->numInputs = numInputs;
            this->numHidden = numHidden;
            this->numOutputs = numOutputs;
        }
};

int main() {
    NeuralNetwork network = NeuralNetwork(3, 3, 1);

    Matrix a = Matrix(2, 3);
    a.randomize(1, 10);
    a.print();
    cout << endl;

    Matrix b = Matrix(3, 2);
    b.randomize(1, 10);
    b.print();
    cout << endl;

    Matrix c = a.matMultiply(b);
    c.print();
}