#include "utils.h"
#include <math.h>
#define BLOCK_SIZE_LENGTH 16
#define BLOCK_SIZE_WIDTH 12
#define THREAD_PER_SM 192
#define SM_NUM 2

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage, 
                        unsigned char* const greyImage, 
                        int numRows, int numCols)
{
    size_t absolute_position_x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t absolute_position_y = blockDim.y * blockIdx.y + threadIdx.y;
    if ( absolute_position_x >= numCols || absolute_position_y >= numRows) return;

    const int index = absolute_position_y * numCols + absolute_position_x;
    uchar4 rgba = rgbaImage[index];
    unsigned char grey = static_cast<unsigned char>(rgba.x * .299f + rgba.y * .587f + rgba.z * .114f);
    greyImage[index] = grey;

    // const int index = THREAD_PER_SM * blockIdx.x + threadIdx.x;
    // uchar4 rgba = rgbaImage[index];
    // unsigned char grey = static_cast<unsigned char>(rgba.x * .299f + rgba.y * .587f + rgba.z * .114f);
    // greyImage[index] = grey;
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
//You must fill in the correct sizes for the blockSize and gridSize
//currently only one block with one thread is being launched
    // size_t block_num_x = numRows / WIDE_RECTANGE + (numRows % WIDE_RECTANGE) == 0 ? 0 : 1;
    // size_t block_num_y = numCols / LONG_RECTANGE + (numCols % LONG_RECTANGE) == 0 ? 0 : 1;

    // const dim3 blockSize(block_num_x, block_num_y, 1);  //TODO
    // const dim3 gridSize( 1, 1, 1);  //TODO
    // rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
    // // *d_greyImage = *d_rgbaImage;
    // cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    const dim3 blockSize( BLOCK_SIZE_LENGTH, BLOCK_SIZE_WIDTH );
    int   grid_size_x = ( numCols + blockSize.x - 1 ) / blockSize.x;
    int   grid_size_y = ( numRows + blockSize.y - 1 ) / blockSize.y; //TODO
    const dim3 gridSize( grid_size_x, grid_size_y );  //TODO
    rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // const int thread = 16;
    // const dim3 blockSize( thread, thread, 1);
    // const dim3 gridSize( ceil(numRows/(float)thread), ceil(numCols/(float)thread), 1);
    // rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

    // cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // rgba_to_greyscale<<<(numRows * numCols + THREAD_PER_SM - 1) / THREAD_PER_SM, THREAD_PER_SM>>>(d_rgbaImage, d_greyImage, numRows, numCols);

    // cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
