#ifndef MATRIX_H
#define MATRIX_H

using namespace std;

class Matrix {
    private:
        vector<vector<float>> cells;

    public:
        int rows;
        int cols;

        Matrix(int rows, int cols);
        Matrix(vector<float> array, bool asRow = false);

        float get(int row, int col) const;
        void set(int row, int col, float value);
        void randomize();

        void add(float n);
        void add(const Matrix& other);
        void multiply(float n);
        void multiply(const Matrix& other);

        static Matrix multiply(const Matrix& a, const Matrix& b);
        static Matrix transpose(const Matrix& matrix);

        void map(const function<float(float)>& fn);
        void map(const function<float(float, int, int)>& fn);
        static Matrix map(const Matrix& matrix, const function<float(float)>& fn);
        static Matrix map(const Matrix& matrix, const function<float(float, int, int)>& fn);

        void print();
        vector<vector<float>> toArray() const;
};

#endif // MATRIX_H
