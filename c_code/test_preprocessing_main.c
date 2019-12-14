#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "preprocess.h"
#include "nn.h"



extern uint8_t test_image[];



int main(void) {

    // Preprocess
    double* output_buf = (double*) malloc(PREPROC_OUTPUT_BUF_SZ*sizeof(double));
    preprocess_camera_image(test_image, output_buf);

    // Write results to file
    FILE *fp;
    fp = fopen("output_image_test.txt", "w");
    for (int i = 0; i < PREPROC_OUTPUT_BUF_SZ; i++) {
        if (i != 0) {
            fprintf(fp, ", ");
        }
        fprintf(fp, "%.2f", output_buf[i]);
    }
    fclose(fp);

    // Run NN on preprocessed image
    matrix_t inpt_mat = {
        .num_rows = 784,
        .num_cols = 1,
        .data = output_buf
    };
    int answr = evaluate_neural_net(inpt_mat);
    printf("Answer: %d\n", answr);

}


