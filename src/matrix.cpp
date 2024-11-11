#include <iostream>
#include <random>

using namespace std;

class Matrix {
    private:
        vector<vector<float>> cells;

    public: 
        int rows;
        int cols;

        Matrix(int rows, int cols) {
            this->rows = rows;
            this->cols = cols;

            // initialize the cells with zeros
            cells = vector<vector<float>>(rows, vector<float>(cols, 0.0f));
        }

        // get cell value at given position
        float get(int row, int col) const {
            return cells[row][col];
        }

        // set cell value at given position
        void set(int row, int col, float value) {
            cells[row][col] = value;
        }

        // set cells to random values in range
        void randomize(float rangeMin, float rangeMax) {
            // seed random engine with device
            random_device rd;
            mt19937 gen(rd());

            // define range for the random numbers
            uniform_real_distribution<> dist(rangeMin, rangeMax);
    
            // set random values at each cell
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    cells[i][j] = floor(dist(gen));
                }
            }
        }

        // add each element by a scalar value
        void add(int n) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    cells[i][j] += n;
                }
            }
        }

        // multiply each element by a scalar value
        void multiply(int n) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    cells[i][j] *= n;
                }
            }
        }
        
        // add another matrix element-wise
        void add(const Matrix& other) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    cells[i][j] += other.get(i, j);
                }
            }
        }

        // multiply another matrix element-wise (hadamard)
        void multiply(const Matrix& other) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    cells[i][j] *= other.get(i, j);
                }
            }
        }

        // matrix multiplication
        Matrix matMultiply(const Matrix& other) {
            Matrix& a = *this;
            Matrix b = other;

            if (a.cols != b.rows) {
                throw invalid_argument("Number of columns of this must match number of rows of other.");
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

        void print() {
            for (int i = 0; i < rows; i++) {
                cout << "[ ";
                for (int j = 0; j < cols; j++) {
                    cout << cells[i][j] << " ";
                }
                cout << "]" << endl;
            }
        }
};
