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
    a.randomize();
    a.print();
    cout << endl;

    Matrix b = Matrix(3, 2);
    b.randomize();
    b.print();
    cout << endl;

    Matrix c = Matrix::multiply(a, b);
    c.print();
    cout << endl;

    Matrix d = Matrix::transpose(c);
    d.print();
}