#include "crow.h"
#include "crow/middlewares/cors.h"

#include <iostream>
#include <fstream>
#include "nn.h"
#include "matrix.h"

using namespace std;

void loadImages(const string& fileName, vector<vector<float>>& images) {
    ifstream file(fileName, ios::binary);

    uint32_t magicNumber;
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    magicNumber = ntohl(magicNumber);

    int numImages;
    file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    numImages = ntohl(numImages);

    int numRows, numCols;
    file.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
    numRows = ntohl(numRows);
    file.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));
    numCols = ntohl(numCols);

    images.resize(numImages, vector<float>(numRows * numCols));
    for (int i = 0; i < numImages; i++) {
        vector<uint8_t> rawImage(numRows * numCols);
        file.read(reinterpret_cast<char*>(rawImage.data()), numRows * numCols);

        for (int j = 0; j < numRows * numCols; j++) {
            images[i][j] = static_cast<float>(rawImage[j]) / 255.0f;
        }
    }

    file.close();
}

void loadLabels(const string& fileName, vector<vector<float>>& labels) {
    ifstream file(fileName, ios::binary);

    uint32_t magicNumber;
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    magicNumber = ntohl(magicNumber);

    int numLabels;
    file.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
    numLabels = ntohl(numLabels);

    labels.resize(numLabels);
    for (int i = 0; i < numLabels; i++) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));

        vector<float> oneHot(10, 0.0f);
        oneHot[label] = 1.0f;
        labels[i] = oneHot;
    }

    file.close();
}

void saveNetwork(const NeuralNetwork& network, string& fileName) {
    ofstream file(fileName, ios::binary);
    vector<char> networkData = network.serialize();
    file.write(networkData.data(), networkData.size());
    file.close();
}

NeuralNetwork loadNetwork(const string& fileName) {
    ifstream file(fileName, ios::binary);
    vector<char> networkData((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();
    return NeuralNetwork(networkData);
}

void train(NeuralNetwork& network, vector<vector<float>>& images, vector<vector<float>>& labels, float learningRate, int epoch) {
    // shuffle the data order
    int n = images.size();
    vector<int> indices(n);
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }
    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    // train network on training data
    int progress = 0;
    for (int i : indices) {
        network.train(images[i], labels[i]);

        // output the current progress
        progress++;
        float progressPercent = static_cast<int>((static_cast<float>(progress) / n) * 100);
        cout << "\rEpoch " << epoch << " | Training " << progressPercent << "%" << flush;
    }
}

void test(NeuralNetwork& network, vector<vector<float>>& images, vector<vector<float>>& labels) {
    // test network on test data
    int n = images.size();
    int correct = 0;
    for (int i = 0; i < n; i++) {
        // get the prediction from the network
        vector<float> prediction = network.predict(images[i]);

        // compare prediction to the correct label (argmax of the labels)
        int predictedLabel = max_element(prediction.begin(), prediction.end()) - prediction.begin();
        int correctLabel = max_element(labels[i].begin(), labels[i].end()) - labels[i].begin();
        if (predictedLabel == correctLabel) {
            correct++;
        }
    }

    // output the test results
    float accuracy = static_cast<float>(correct) / n * 100.0f;
    cout <<  " | " << accuracy << "% Accuracy" << endl;
}

void trainAndSaveModel() {
    // set training parameters
    int numHidden = 256;
    int numEpochs = 10;
    float learningRate = 0.05;
    
    // load training & testing data from file
    string trainImageFile = "data/train-images.idx3-ubyte";
    string trainLabelFile = "data/train-labels.idx1-ubyte";
    string testImageFile = "data/t10k-images.idx3-ubyte";
    string testLabelFile = "data/t10k-labels.idx1-ubyte";

    vector<vector<float>> trainImages, testImages;
    vector<vector<float>> trainLabels, testLabels;

    loadImages(trainImageFile, trainImages);
    loadLabels(trainLabelFile, trainLabels);
    loadImages(testImageFile, testImages);
    loadLabels(testLabelFile, testLabels);

    // define neural network architecture
    int numInputs = trainImages[0].size(); // 784
    int numOutputs = trainLabels[0].size(); // 10
    NeuralNetwork network = NeuralNetwork(numInputs, numHidden, numOutputs);

    // train and test accuracy at each epoch
    cout << "Neural network with " << numHidden << " hidden neurons and " << learningRate << " learning rate..." << endl;
    for (int epoch = 1; epoch <= numEpochs; epoch++) {
        train(network, trainImages, trainLabels, learningRate, epoch);
        test(network, testImages, testLabels);
    }

    // save network
    string modelOutputFile = "model/model.bin";
    saveNetwork(network, modelOutputFile);
    cout << "Model successfully saved to " << modelOutputFile << endl;
}

void runPredictionService() {
    // load network
    NeuralNetwork network = loadNetwork("model/model.bin");

    // Create app with CORS middleware
    crow::App<crow::CORSHandler> app;
    
    // Configure CORS
    auto& cors = app.get_middleware<crow::CORSHandler>();
    cors
        .global()
        .headers("Content-Type", "Accept")
        .methods("POST"_method, "OPTIONS"_method)
        .origin("*");  // Allow any origin
    
    // Define prediction endpoint
    CROW_ROUTE(app, "/predict").methods("POST"_method)
    ([&network](const crow::request& req) {
        // receive image pixel data
        auto body = req.body;
        auto json = crow::json::load(body);
        vector<float> input;
        for (const auto& val : json["image"]) {
            input.push_back(val.d());
        }
        
        // Run prediction
        vector<float> prediction = network.predict(input);
        int number = max_element(prediction.begin(), prediction.end()) - prediction.begin();
        return crow::response(to_string(number));
    });
    
    // start the HTTP server
    app.port(8080).run();
}

int main(int argc, char* argv[]) {
    string operation = argv[1];

    if (operation == "train") {
        trainAndSaveModel();
    } else if (operation == "run"){
        runPredictionService();
    }

    return 0;
}
