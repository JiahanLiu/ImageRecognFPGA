#include <stdbool.h>
#include "preprocess.h"



#define THRESHOLD           (0.20 * 255)

// Cropping parameters
#define CROP_MARGIN         0.85
#define TARGET_SIZE         (CROP_MARGIN*CAMERA_IMAGE_HEIGHT)
#define LEFTRIGHT_MARGIN    ((int)((CAMERA_IMAGE_WIDTH-TARGET_SIZE)/2.0))
#define TOPBOTTOM_MARGIN    ((int)((CAMERA_IMAGE_HEIGHT-TARGET_SIZE)/2.0))
#define CROP_LEFT_EDGE      LEFTRIGHT_MARGIN
#define CROP_RIGHT_EDGE     (CAMERA_IMAGE_WIDTH-LEFTRIGHT_MARGIN)
#define CROP_TOP_EDGE       TOPBOTTOM_MARGIN
#define CROP_BOTTOM_EDGE    (CAMERA_IMAGE_HEIGHT-TOPBOTTOM_MARGIN)

#define BLOCK_SIZE          ((int)(TARGET_SIZE/28.0) + 1)


bool check_pixel_threshold(uint8_t* camera_image, int r, int c);


void preprocess_camera_image(uint8_t* camera_image, double* output_buf) {
    // Iterate over the blocks that map to downsamples in the output
    for (int rblock = 0; rblock < 28; rblock++) {
        for (int cblock = 0; cblock < 28; cblock++) {
            int blk_left_edge = CROP_LEFT_EDGE + cblock*BLOCK_SIZE;
            int blk_right_edge = blk_left_edge + BLOCK_SIZE;
            int blk_top_edge = CROP_TOP_EDGE + rblock*BLOCK_SIZE;
            int blk_bottom_edge = blk_top_edge + BLOCK_SIZE;
            // Iterate over the pixels within this block
            double blk_value = 0.0;
            bool found_max = false;
            for (int r = blk_top_edge; r < blk_bottom_edge && r < CROP_BOTTOM_EDGE && !found_max; r++) {
                for (int c = blk_left_edge; c < blk_right_edge && c < CROP_RIGHT_EDGE && !found_max; c++) {
                    if(check_pixel_threshold(camera_image, r, c)) {
                        // since we are taking the max, we can stop here
                        blk_value = 1.0;
                        found_max = true;
                    }
                }
            }
            // Save downsample result to output
            int idx = cblock + rblock*28;
            output_buf[idx]  = blk_value;
        }
    }
}


bool check_pixel_threshold(uint8_t* camera_image, int r, int c) {
    int pixel_sz = 3*sizeof(uint8_t);
    int idx = (pixel_sz*c) + (pixel_sz*r)*CAMERA_IMAGE_WIDTH;
    uint8_t red = camera_image[idx + 0];
    uint8_t green = camera_image[idx + 1];
    uint8_t blue = camera_image[idx + 2];
    double avg = ((double)(red+green+blue)) / 3.0;
    if (avg < THRESHOLD) {
        return true;
    } else {
        return false;
    }
}

