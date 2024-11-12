#include <iostream>
#include <random>

using namespace std;

class Matrix {
    private:
        vector<vector<float>> cells;

    public: 
        int rows;
        int cols;

        // initialize matrix with zeros
        Matrix(int rows, int cols) {
            this->rows = rows;
            this->cols = cols;
            cells = vector<vector<float>>(rows, vector<float>(cols, 0.0f)); // init with zeros
        }

        // Initialize a matrix from a 1D vector
        Matrix(vector<float> array, bool asRow = false) {
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
        float get(int row, int col) const {
            return cells[row][col];
        }

        // set cell value at given position
        void set(int row, int col, float value) {
            cells[row][col] = value;
        }

        // set cells to random values in (-1, 1)
        void randomize() {
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
        void add(float n) {
            map([n](float x) { return x + n; });
        }

        // multiply a scalar value to every element in matrix
        void multiply(float n) {
            map([n](float x) { return x * n; });
        }
        
        // element-wise add another matrix
        void add(const Matrix& other) {
            map([&other](float x, int i, int j) { return x + other.get(i, j); });
        }

        // element-wise multiply another matrix (hadamard)
        void multiply(const Matrix& other) {
            map([&other](float x, int i, int j) { return x * other.get(i, j); });
        }

        // static matrix multiplication: a x b
        static Matrix multiply(const Matrix& a, const Matrix& b) {
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
        static Matrix transpose(const Matrix& matrix) {
            Matrix result = Matrix(matrix.cols, matrix.rows);
            result.map([&matrix](float x, int i, int j) { return matrix.get(j, i); });
            return result;
        }

        // map function: apply a function fn(value) to every element in matrix
        void map(const function<float(float)>& fn) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    cells[i][j] = fn(cells[i][j]);
                }
            }
        }

        // map function: apply a function fn(value, i, j) to every element in matrix
        void map(const function<float(float, int, int)>& fn) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    cells[i][j] = fn(cells[i][j], i, j);
                }
            }
        }

        // static map function: apply a function fn(value) to every element in matrix
        static Matrix map(const Matrix& matrix, const function<float(float)>& fn) {
            Matrix result = Matrix(matrix.rows, matrix.cols);
            result.map([&matrix, &fn](float x, int i, int j) { return fn(matrix.get(i, j)); });
            return result;
        }
        
        // map function: apply a function fn(value, i, j) to every element in matrix
        static Matrix map(const Matrix& matrix, const function<float(float, int, int)>& fn) {
            Matrix result = Matrix(matrix.rows, matrix.cols);
            result.map([&matrix, &fn](float x, int i, int j) { return fn(matrix.get(i, j), i, j); });
            return result;
        }

        // display matrix
        void print() {
            for (int i = 0; i < rows; i++) {
                cout << "[ ";
                for (int j = 0; j < cols; j++) {
                    cout << cells[i][j] << " ";
                }
                cout << "]" << endl;
            }
        }

        // get copy of cells
        vector<vector<float>> toArray() const {
            return cells;
        }
};
