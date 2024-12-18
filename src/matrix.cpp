#include <iostream>
#include <random>
#include "matrix.h"
#include <cstring>

using namespace std;

// initialize matrix with zeros
Matrix::Matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    cells = vector<vector<float>>(rows, vector<float>(cols, 0.0f)); // init with zeros
}

// Initialize a matrix from a 1D vector
Matrix::Matrix(vector<float> array, bool asRow) {
    if (asRow) {
        // Treat the vector as a single row
        rows = 1;
        cols = array.size();
        cells = vector<vector<float>>(1, array);
    } else {
        // Treat the vector as a single column
        rows = array.size();
        cols = 1;
        cells = vector<vector<float>>(rows, vector<float>(1));
        for (int i = 0; i < rows; i++) {
            cells[i][0] = array[i];
        }
    }
}

// get cell value at given position
float Matrix::get(int row, int col) const {
    return cells[row][col];
}

// set cell value at given position
void Matrix::set(int row, int col, float value) {
    cells[row][col] = value;
}

// set cells to random values in (-1, 1)
void Matrix::randomize() {
    // seed random engine with device
    random_device rd;
    mt19937 gen(rd());

    // define range for the random numbers
    uniform_real_distribution<float> dist(-1, 1);

    // set random values at each cell
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cells[i][j] = dist(gen);
        }
    }
}

// add a scalar value to every element in matrix
void Matrix::add(float n) {
    map([n](float x) { return x + n; });
}

// multiply a scalar value to every element in matrix
void Matrix::multiply(float n) {
    map([n](float x) { return x * n; });
}

// element-wise add another matrix
void Matrix::add(const Matrix& other) {
    map([&other](float x, int i, int j) { return x + other.get(i, j); });
}

// element-wise multiply another matrix (hadamard)
void Matrix::multiply(const Matrix& other) {
    map([&other](float x, int i, int j) { return x * other.get(i, j); });
}

// static matrix multiplication: a x b
Matrix Matrix::multiply(const Matrix& a, const Matrix& b) {
    if (a.cols != b.rows) {
        throw invalid_argument("Number of columns in 'a' must match number of rows in 'b'.");
    }

    Matrix result = Matrix(a.rows, b.cols);
    int dim = a.cols; // same as b.rows
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            float sum = 0;
            for (int k = 0; k < dim; k++) {
                sum += a.get(i,k)* b.get(k,j);  // dot product of row i of a and column j of b
            }
            result.set(i, j, sum);
        }
    }
    return result;
}

// static matrix transpose
Matrix Matrix::transpose(const Matrix& matrix) {
    Matrix result = Matrix(matrix.cols, matrix.rows);
    result.map([&matrix](float x, int i, int j) { return matrix.get(j, i); });
    return result;
}

// map function: apply a function fn(value) to every element in matrix
void Matrix::map(const function<float(float)>& fn) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cells[i][j] = fn(cells[i][j]);
        }
    }
}

// map function: apply a function fn(value, i, j) to every element in matrix
void Matrix::map(const function<float(float, int, int)>& fn) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cells[i][j] = fn(cells[i][j], i, j);
        }
    }
}

// static map function: apply a function fn(value) to every element in matrix
Matrix Matrix::map(const Matrix& matrix, const function<float(float)>& fn) {
    Matrix result = Matrix(matrix.rows, matrix.cols);
    result.map([&matrix, &fn](float x, int i, int j) { return fn(matrix.get(i, j)); });
    return result;
}

// map function: apply a function fn(value, i, j) to every element in matrix
Matrix Matrix::map(const Matrix& matrix, const function<float(float, int, int)>& fn) {
    Matrix result = Matrix(matrix.rows, matrix.cols);
    result.map([&matrix, &fn](float x, int i, int j) { return fn(matrix.get(i, j), i, j); });
    return result;
}

// display matrix
void Matrix::print() {
    for (int i = 0; i < rows; i++) {
        cout << "[ ";
        for (int j = 0; j < cols; j++) {
            cout << cells[i][j] << " ";
        }
        cout << "]" << endl;
    }
}

// get copy of cells
vector<vector<float>> Matrix::toArray() const {
    return cells;
}

// serialize matrix
vector<char> Matrix::serialize() const {
    vector<char> serializedData;

    // serialize rows and columns
    const char* numRowsData = reinterpret_cast<const char*>(&rows);
    const char* numColsData = reinterpret_cast<const char*>(&cols);
    serializedData.insert(serializedData.end(), numRowsData, numRowsData + sizeof(rows));
    serializedData.insert(serializedData.end(), numColsData, numColsData + sizeof(cols));

    // serialize matrix data (row by row)
    for (const vector<float>& row : cells) {
        const char* rowData = reinterpret_cast<const char*>(row.data());
        serializedData.insert(serializedData.end(), rowData, rowData + cols * sizeof(float));
    }

    return serializedData;
}

// initialize matrix from serialized data
Matrix::Matrix(const vector<char>& serializedData) {
    const char* dataPtr = serializedData.data();
    
    // deserialize rows and columns
    rows = *reinterpret_cast<const int*>(dataPtr);
    dataPtr += sizeof(rows);
    cols = *reinterpret_cast<const int*>(dataPtr);
    dataPtr += sizeof(cols);
    cells = vector<vector<float>>(rows, vector<float>(cols, 0.0f));

    // deserialize matrix data (row by row)
    for (vector<float>& row : cells) {
        memcpy(row.data(), dataPtr, cols * sizeof(float));
        dataPtr += cols * sizeof(float);
    }
}