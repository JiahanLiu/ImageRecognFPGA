#ifndef _nn_h_
#define _nn_h_

#define NN_INPUT_SIZE   784 // 28*28
#define NN_OUTPUT_SIZE  10 // 10 digits
#define LAYER_WIDTH_PLACEHOLDER  1000000

extern int num_hidden_layers;
extern int width_hidden_layers;
extern double hidden_layer_weights[][LAYER_WIDTH_PLACEHOLDER];
extern double hidden_layer_biases[][LAYER_WIDTH_PLACEHOLDER];
extern double output_layer_weight[];
extern double output_layer_bias[];

typedef struct {
    int num_rows;
    int num_cols;
    double* data;
} matrix_t;

int evaluate_neural_net(matrix_t input);

#endif // _nn_h_

