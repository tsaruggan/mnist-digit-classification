#include <iostream>
#include <random>
#include <vector>

using namespace std;

// My target line function
float f(const int x) {
    // y = mx + b
    return 0.5 * x + 0.2;
}

// Activation function
int sign(float n) {
    if (n >= 0) {
        return 1;
    } else {
        return -1;
    }
}

// Generate a random number within a range
float random(float rangeMin, float rangeMax) {
    // Create a random device and a Mersenne Twister random engine
    random_device rd;  // Seed for random number engine
    mt19937 gen(rd()); // Mersenne Twister engine seeded with rd

    // Define range for the random number (-1, 1)
    uniform_real_distribution<> dist(rangeMin, rangeMax);

    // Generate random number
    return dist(gen);
}

class Point {
    public:
        float x;
        float y;
        int label;

        Point() {
            x = random(-1, 1);
            y = random(-1, 1);
            
            float lineY = f(x);
            if (y >= lineY) {
                label = 1;
            } else {
                label = -1;
            }
        }
};

class Perceptron {
    vector<float> weights;
    float bias;
    int numWeights;
    float learningRate;

    public: 
        Perceptron(const int numWeights, const float learningRate=0.1) {
            this->numWeights = numWeights;
            this->learningRate = learningRate;
            
            // Initialize weights & bias randomly
            for (int i = 0; i < numWeights; i++) {
                weights.push_back(random(-1, 1));
            }
            bias = random(-1, 1);
        }

        float guessY(float x) {
            float w0 = weights[0];  // weight for x
            float w1 = weights[1];  // weight for y
            float w2 = bias;        // bias term
            return -(w0 / w1) * x - (w2 / w1);
        }

        int predict(vector<float> inputs) {
            float sum = 0;
            for (int i = 0; i < numWeights; i++) {
                sum += inputs[i] * weights[i];
            }
            sum += bias;
            int output = sign(sum);
            return output;
        }

        void train(vector<float> inputs, int target) {
            int prediction = predict(inputs);
            int error = target - prediction; 

            for (int i = 0; i < numWeights; i++) {
                weights[i] += error * inputs[i] * learningRate;
            }
            bias += error * learningRate;
        }
};

// int main() {
//     // Initialize points
//     vector<Point*> points;
//     int numPoints = 50;
//     for (int i = 0; i < numPoints; i++) {
//         Point* point = new Point();
//         points.push_back(point);
//     }

//     // Initialize perceptron model
//     Perceptron perceptron(2, 0.5);

//     // Train
//     int epochs = 10000;
//     for (int epoch = 0; epoch < epochs; epoch++) {
//         for (Point* point : points) {
//             vector<float> inputs = {point->x, point->y};
//             int target = point->label;
//             perceptron.train(inputs, target);
//         }
//     }
    
//     // Evaluate
//     for (Point* point : points) {
//         vector<float> inputs = {point->x, point->y};
//         int prediction = perceptron.predict(inputs);
//         float guessY = perceptron.guessY(point->x);

//         string correct;
//         if (prediction == point->label) {
//             correct = "Correct";
//         } else {
//             correct = "Incorrect";
//         }
//         cout << "(" << point->x << "," << point->y << ") => " << guessY << ":" << correct << endl;
//     }

//     float x1 = -1.0;
//     float x2 = 1.0;
//     float y1 = perceptron.guessY(x1);
//     float y2 = perceptron.guessY(x2);
//     float m = (y2-y1) / (x2-x1);
//     float b = y2 - m * x2;
//     cout << "Line: " << "y = " << m << "x + " << b << endl;
    
//     // Clean up
//     for (Point* point : points) {
//         delete point;
//     }
//     return 0;
// }
