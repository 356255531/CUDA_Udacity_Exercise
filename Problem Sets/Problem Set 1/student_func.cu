#include "utils.h"
#define LONG_RECTANGE 16
#define WIDE_RECTANGE 12
#define THREAD_PER_SM 192
#define SM_NUM 2

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage, 
                        unsigned char* const greyImage, 
                        int numRows, int numCols)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    size_t j = blockDim.y * threadIdx.y + threadIdx.y;
    if ( i >= numRows || j >= numCols)
    {
        return;
    }

    uchar4 rgba = rgbaImage[i + j * numCols];

    greyImage[i + j * numCols] = rgba.x * .299f + rgba.y * .587f + rgba.z * .114f;
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
//You must fill in the correct sizes for the blockSize and gridSize
//currently only one block with one thread is being launched
    size_t block_num_x = numRows / WIDE_RECTANGE + (numRows % WIDE_RECTANGE) == 0 ? 0 : 1;
    size_t block_num_y = numCols / LONG_RECTANGE + (numCols % LONG_RECTANGE) == 0 ? 0 : 1;

    const dim3 blockSize(block_num_x, block_num_y, 1);  //TODO
    const dim3 gridSize( 1, 1, 1);  //TODO
    rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
    // *d_greyImage = *d_rgbaImage;
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}
