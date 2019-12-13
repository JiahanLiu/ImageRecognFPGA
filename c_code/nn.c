#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "nn.h"



#define MAT(m, r, c)  m.data[r*m.num_cols + c]


matrix_t matmul(matrix_t a, matrix_t b, double* dest_buf);
matrix_t vecadd(matrix_t a, matrix_t b, double* dest_buf);
void relu(matrix_t a);
int vecargmax(matrix_t a);

#ifdef DEBUGNN
void print_vec(matrix_t v);
void print_vec_n(matrix_t v, int max_num);
#endif




int evaluate_neural_net(matrix_t input) {
    int largest_array = width_hidden_layers;
    double* buf1 = (double*) malloc(largest_array*sizeof(double));
    double* buf2 = (double*) malloc(largest_array*sizeof(double));
    matrix_t arr = input;
    // hidden layers
    matrix_t weight_mat;
    weight_mat.num_rows = width_hidden_layers;
    weight_mat.num_cols = NN_INPUT_SIZE;
    matrix_t bias_mat;
    bias_mat.num_rows = width_hidden_layers;
    bias_mat.num_cols = 1;
    for (int h=0; h < num_hidden_layers; h++) {
        // multiply weights
        weight_mat.data = hidden_layer_weights[h];
        arr = matmul(weight_mat, arr, buf1);
        weight_mat.num_cols = width_hidden_layers;
        // add bias
        bias_mat.data = hidden_layer_biases[h];
        arr = vecadd(bias_mat, arr, buf2);
        // ReLU
        relu(arr);
    }
    // output layers
    weight_mat.num_rows = NN_OUTPUT_SIZE;
    weight_mat.num_cols = width_hidden_layers;
    weight_mat.data = output_layer_weight;
    bias_mat.num_rows = NN_OUTPUT_SIZE;
    bias_mat.num_cols = 1;
    bias_mat.data = output_layer_bias;
    arr = matmul(weight_mat, arr, buf1);
    arr = vecadd(bias_mat, arr, buf2);
#ifdef DEBUGNN
    printf("NN output before argmax:\n");
    print_vec(arr);
#endif
    // return max
    int argmax = vecargmax(arr);
    free(buf1);
    free(buf2);
    return argmax;
}


matrix_t matmul(matrix_t a, matrix_t b, double* dest_buf) {
    assert(a.num_cols == b.num_rows);
    matrix_t ret;
    ret.num_rows = a.num_rows;
    ret.num_cols = b.num_cols;
    ret.data = dest_buf;
    for (int r = 0; r < ret.num_rows; r++) {
        for (int c = 0; c < ret.num_cols; c++) {
            double sum = 0;
            for (int k = 0; k < a.num_cols; k++) {
                sum += MAT(a,r,k)*MAT(b,k,c);
            }
            MAT(ret,r,c) = sum;
        }
    }
    return ret;
}


matrix_t vecadd(matrix_t a, matrix_t b, double* dest_buf) {
    assert(a.num_rows == b.num_rows);
    assert(a.num_cols == b.num_cols);
    matrix_t ret;
    ret.num_rows = a.num_rows;
    ret.num_cols = a.num_cols;
    ret.data = dest_buf;
    for (int r = 0; r < a.num_rows; r++) {
        for (int c = 0; c < a.num_cols; c++) {
            MAT(ret,r,c) = MAT(a,r,c) + MAT(b,r,c);
        }
    }
    return ret;
}


void relu(matrix_t a) {
    for (int r = 0; r < a.num_rows; r++) {
        for (int c = 0; c < a.num_cols; c++) {
            if (MAT(a, r, c) < 0) {
                MAT(a, r, c) = 0;
            }
        }
    }
}


int vecargmax(matrix_t a) {
    assert(a.num_cols == 1);
    double max = MAT(a, 0, 0);
    int max_idx = 0;
    for (int r = 0; r < a.num_rows; r++) {
        if (max < MAT(a, r, 0)) {
            max = MAT(a, r, 0);
            max_idx = r;
        }
    }
    return max_idx;
}


#ifdef DEBUGNN
void print_vec_n(matrix_t v, int max_num) {
    assert(v.num_cols == 1);
    printf("<");
    int r;
    for (r = 0; r < v.num_rows && r != max_num; r++) {
        if (r != 0) {
            printf(", ");
        }
        printf("%f", MAT(v, r, 0));
    }
    if (r == max_num) {
        printf(", ...");
    }
    printf(">");
    printf("\n");
}

void print_vec(matrix_t v) {
    print_vec_n(v, -1);
}
#endif
