#include <iostream>
#include <random>
#include <vector>

using namespace std;

// Activation function
int sign(double n) {
    if (n >= 0) {
        return 1;
    } else {
        return -1;
    }
}

// Generate a random number within a range
double random(double rangeMin, double rangeMax) {
    // Create a random device and a Mersenne Twister random engine
    random_device rd;  // Seed for random number engine
    mt19937 gen(rd()); // Mersenne Twister engine seeded with rd

    // Define range for the random number (-1, 1)
    uniform_int_distribution<> dist(rangeMin, rangeMax);

    // Generate random number
    return dist(gen);
}

class Point {
    public:
        double x;
        double y;
        int label;

        Point() {
            x = random(0, 10);
            y = random(0, 10);

            // y = x line
            if (x > y) {
                label = 1;
            } else {
                label = -1;
            }
        }
};

class Perceptron {
    vector<double> weights;
    int numWeights;
    double learningRate;

    public: 
        Perceptron(const int numWeights, const double learningRate=0.1) {
            this->numWeights = numWeights;
            this->learningRate = learningRate;
            
            // Initialize weights randomly
            for (int i = 0; i < numWeights; i++) {
                weights.push_back(random(-1, 1));
            }
        }

        int guess(vector<double> inputs) {
            double sum = 0;
            for (int i = 0; i < numWeights; i++) {
                sum += inputs[i] * weights[i];
            }
            int output = sign(sum);
            return output;
        }

        void train(vector<double> inputs, int target) {
            int prediction = guess(inputs);
            int error = target - prediction; 

            for (int i = 0; i < numWeights; i++) {
                weights[i] += error * inputs[i] * learningRate;
            }
        }
};

int main() {
    vector<Point*> points;
    int numPoints = 100;
    for (int i = 0; i < 100; i++) {
        Point* point = new Point();
        points.push_back(point);
    }

    // Train & eval
    Perceptron perceptron(2);
    for (Point* point : points) {
        vector<double> inputs = {point->x, point->y};
        int target = point->label;

        perceptron.train(inputs, target);

        int guess = perceptron.guess(inputs);
        if (guess == target) {
            cout << "Correct" << endl;
        } else {
            cout << "Incorrect" << endl;
        }
    }

    // Clean up
    for (Point* point : points) {
        delete point;
    }
    return 0;
}
