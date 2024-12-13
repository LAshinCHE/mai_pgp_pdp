#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <chrono> 

void SwapRows(std::vector<std::vector<double>>& matrix, int row1, int row2) {
    std::swap(matrix[row1], matrix[row2]);
}

void ZeroingLowerColumns(std::vector<std::vector<double>>& matrix, int n, int i) {
    for (int row = i + 1; row < n; ++row) {
        double factor = -matrix[row][i] / matrix[i][i];
        for (int col = i; col < 2 * n; ++col) {
            matrix[row][col] += matrix[i][col] * factor;
        }
    }
}

void ZeroingUpperColumns(std::vector<std::vector<double>>& matrix, int n, int i) {
    for (int row = 0; row < i; ++row) {
        double factor = -matrix[row][i] / matrix[i][i];
        for (int col = i; col < 2 * n; ++col) {
            matrix[row][col] += matrix[i][col] * factor;
        }
    }
}

void Normalize(std::vector<std::vector<double>>& matrix, int n) {
    for (int i = 0; i < n; ++i) {
        double divisor = matrix[i][i];
        for (int j = n; j < 2 * n; ++j) {
            matrix[i][j] /= divisor;
        }
    }
}

void PrintReverseMatrix(const std::vector<std::vector<double>>& matrix, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = n; j < 2 * n; ++j) {
            std::cout << std::fixed << std::setprecision(10) << matrix[i][j] << " ";
        }
        std::cout << "\n";
    }
}

void InvertMatrix(std::vector<std::vector<double>>& matrix, int n) {
    for (int i = 0; i < n; ++i) {
        int maxIndex = i;
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(matrix[k][i]) > std::abs(matrix[maxIndex][i])) {
                maxIndex = k;
            }
        }
        if (maxIndex != i) {
            SwapRows(matrix, i, maxIndex);
        }
        ZeroingLowerColumns(matrix, n, i);
    }

    for (int i = n - 1; i >= 0; --i) {
        ZeroingUpperColumns(matrix, n, i);
    }

    Normalize(matrix, n);
}

int main() {
    int n;
    std::cin >> n;

    std::vector<std::vector<double>> matrix(n, std::vector<double>(2 * n, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> matrix[i][j];
        }
    }

    for (int i = 0; i < n; ++i) {
        matrix[i][n + i] = 1.0;
    }

    auto start = std::chrono::high_resolution_clock::now();

    InvertMatrix(matrix, n);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    PrintReverseMatrix(matrix, n);

    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
