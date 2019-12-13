#ifndef _preprocess_h_
#define _preprocess_h_

#include <stdint.h>


#define PREPROC_OUTPUT_BUF_SZ   784 // 28*28
#define CAMERA_IMAGE_WIDTH      1280
#define CAMERA_IMAGE_HEIGHT     720


/*
 * Preprocessing:
 * Input: Expects a pointer to the image stored as follows. Each pixel is
 *        24-bits, 1 byte for red, then green, then blue. Pixels are
 *        ordered row 0 col 0, row 0 col 1, row 0 col 2, etc. The image
 *        is of size #define'd above.
 * Processing: The image will be cropped, converted to grayscale,
 *             thresholded, and downsampled to 28x28.
 * Output: Will write the downsampled image as 28x28 doubles.(though each
 *         entry will be either 0. or 1.)
 */
void preprocess_camera_image(uint8_t* camera_image, double* output_buf);


#endif // _preprocess_h_
