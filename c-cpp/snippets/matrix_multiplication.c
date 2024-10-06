#include <stdio.h>
#include <stdlib.h>

// Function to dynamically allocate a matrix
int** allocate_matrix(int rows, int cols) {
    int **matrix = (int **)malloc(rows * sizeof(int *));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int *)malloc(cols * sizeof(int));
    }
    return matrix;
}

// Function to free the memory of a matrix
void free_matrix(int **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Function to multiply two matrices
int** multiply_matrices(int **a, int **b, int rows_a, int cols_a, int cols_b) {
    // Dynamically allocate result matrix
    int **result = allocate_matrix(rows_a, cols_b);

    // Loop to multiply matrices
    for (int i = 0; i < rows_a; i++) {
        for (int j = 0; j < cols_b; j++) {
            result[i][j] = 0;
            for (int k = 0; k < cols_a; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}

// Function to input the elements of a matrix
void input_matrix(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("Enter element [%d][%d]: ", i, j);
            scanf("%d", &matrix[i][j]);
        }
    }
}

// Function to print a matrix
void print_matrix(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main() {
    int rows_a, cols_a, rows_b, cols_b;

    // Input dimensions for matrix A
    printf("Enter the number of rows for matrix A: ");
    scanf("%d", &rows_a);
    printf("Enter the number of columns for matrix A: ");
    scanf("%d", &cols_a);

    // Input dimensions for matrix B
    printf("Enter the number of rows for matrix B: ");
    scanf("%d", &rows_b);
    printf("Enter the number of columns for matrix B: ");
    scanf("%d", &cols_b);

    // Check if matrices can be multiplied
    if (cols_a != rows_b) {
        printf("Error: The number of columns in matrix A must be equal to the number of rows in matrix B.\n");
        return -1;
    }

    // Dynamically allocate matrices A and B
    int **a = allocate_matrix(rows_a, cols_a);
    int **b = allocate_matrix(rows_b, cols_b);

    // Input elements of matrices
    printf("Enter the elements of matrix A:\n");
    input_matrix(a, rows_a, cols_a);
    
    printf("Enter the elements of matrix B:\n");
    input_matrix(b, rows_b, cols_b);

    // Multiply matrices
    int **result = multiply_matrices(a, b, rows_a, cols_a, cols_b);

    // Print the result
    printf("Resulting matrix after multiplication:\n");
    print_matrix(result, rows_a, cols_b);

    // Free dynamically allocated memory
    free_matrix(a, rows_a);
    free_matrix(b, rows_b);
    free_matrix(result, rows_a);

    return 0;
}
