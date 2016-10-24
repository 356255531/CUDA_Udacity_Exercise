/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#define BLOCK_LENGTH 16
#define BLOCK_WIDTH 12
#define NUM_SM 2
#define NUM_THREAD 192


void find_extrem_value(const float* const d_logLuminance, 
                       float &min_logLum, 
                       float &max_logLum, 
                       const size_t numRows, 
                       const size_t numCols);

__global__
void _k_schared_reduce_min_max(const float* const d_input, 
                               float* const d_output, 
                               const size_t input_size, 
                               const bool min_or_max);


unsigned int* compute_histogram(const float* const d_logLuminance, 
                          const float &min_logLum, 
                          const float &lum_range, 
                          const size_t numBins, 
                          const size_t numRows, 
                          const size_t numCols);

__global__
unsigned int* _k_schared_generate_histogram(const float* const d_logLuminance, 
                                            const float &min_logLum, 
                                            const float &lum_range, 
                                            unsigned int* const d_histogram, 
                                            const size_t input_size);

__global__
void _k_schared_reduce_histogram(const float* const d_histogram_input, 
                                 unsigned int* const d_histogram, 
                                 const size_t input_size, 
                                 const size_t numBins);

__global__
void _k_perform_scan(const unsigned int* const d_histogram, 
                     unsigned int* const d_cdf, 
                     size_t const numBins);



void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

// 1) find the minimum and maximum value in the input logLuminance channel
//    store in min_logLum and max_logLum

  find_extrem_value(d_logLuminance, min_logLum, max_logLum,numRows, numCols);

  // 2) subtract them to find the range
  float lum_range = (max_logLum - min_logLum) / (numBins * 1.0);
  // 3) generate a histogram of all the values in the logLuminance channel using
  // the formula: bin = (lum[i] - lumMin) / lumRange * numBins

  unsigned int* d_histogram = compute_histogram(d_luminance, min_logLum, lum_range, 
                                          numBins, numRows, numCols);

  // 4) Perform an exclusive scan (prefix sum) on the histogram to get
  // the cumulative distribution of luminance values (this should go in the
  // incoming d_cdf pointer which already has been allocated for you) 

  dim3 grid_size((numBins + NUM_THREAD - 1) / NUM_THREAD),
       block_size(NUM_THREAD);

  _k_perform_scan<<<grid_size, block_size>>>(d_histogram, d_cdf, numBins);

  checkCudaErrors(cudaFree(d_histogram));
  d_histogram = NULL;
}

void find_extrem_value(const float* const d_logLuminance, 
                       float &min_logLum, 
                       float &max_logLum, 
                       const size_t numRows, 
                       const size_t numCols)
{
  size_t input_size = numRows * numCols;
  size_t grid_size = (input_size + NUM_THREAD - 1) / NUM_THREAD, 
         block_size = NUM_THREAD;
  float *d_max_input, *d_min_input, *d_max_output, *d_min_output;

  checkCudaErrors(cudaMemMalloc(&d_min_input, input_size));
  checkCudaErrors(cudaMemMalloc(&d_max_output, input_size));

  checkCudaErrors(cudaMemcpy(d_min_input, d_logLuminance, 
                            input_size * sizeof(float), 
                            cudaMemcpyDeviceToDevice));

  checkCudaErrors(cudaMemcpy(d_max_output, d_logLuminance, 
                            input_size * sizeof(float), 
                            cudaMemcpyDeviceToDevice));

  while(gridSize > 1) {
    checkCudaErrors(cudaMemMalloc(&d_max_output, sizeof(float) * grid_size));
    checkCudaErrors(cudaMemMalloc(&d_min_output, sizeof(float) * grid_size));

    _k_schared_reduce_min_max<<<grid_size, block_size>>>(d_min_input, 
                                                         d_min_output, 
                                                         input_size, 
                                                         0);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    _k_schared_reduce_min_max<<<grid_size, block_size>>>(d_max_input, 
                                                         d_max_output, 
                                                         input_size, 
                                                         1);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    input_size = grid_size;
    grid_size = (grid_size + NUM_THREAD - 1) / NUM_THREAD;

    checkCudaErrors(cudaFree(d_min_input));
    checkCudaErrors(cudaFree(d_max_input));
    d_min_input = d_min_output;
    d_max_input = d_max_output;
  }

  min_logLum = d_min_input[0];
  max_logLum = d_max_input[0];

  checkCudaErrors(cudaFree(d_max_input));
  checkCudaErrors(cudaFree(d_min_input));
}

__global__
void _k_schared_reduce_min_max(const float* const d_input, 
                               float* const d_output, 
                               const size_t input_size, 
                               const bool min_or_max)
{
  const size_t my_absolute_position = blockDim.x * blockIdx.x + threadIdx.x;
  const int thread_id = threadIdx.x;

  extern __shared__ float shared_input_copy[];
  if (my_absolute_position < input_size) 
    shared_input_copy[thread_id] = d_max_input[my_absolute_position];
  else {
      if (min_or_max) {
        shared_input_copy[thread_id] = FLT_INF;
      }
      else {
        shared_input_copy[thread_id] = -FLT_INF;
      }
      return;
  }

  __syncthreads();

  for (unsigned int i = threadDim.x / 2; i > 0; i>>=1)
  {
    if (thread_id < i)
    {
      if (min_or_max) {
        shared_input_copy[thread_id] = std::min(shared_input_copy[thread_id], 
                                                shared_input_copy[thread_id + i]);
      }
      else {
        shared_input_copy[thread_id] = std::max(shared_input_copy[thread_id], 
                                                  shared_input_copy[thread_id + i]);
      }
    }
  }

  __syncthreads();


  if (thread_id == 0)
  {
    d_output[blockIdx.x] = shared_input_copy[0];
  }

}

unsigned int* compute_histogram(const float* const d_logLuminance, 
                          const float &min_logLum, 
                          const float &lum_range, 
                          const size_t numBins, 
                          const size_t numRows, 
                          const size_t numCols)
{
  size_t input_size = numRows * numCols;
  size_t grid_size = (input_size + NUM_THREAD - 1) / 
                      NUM_THREAD, 
         block_size = NUM_THREAD;
  
  unsigned int *d_histogram;
  checkCudaErrors(cudaMemMalloc(&d_histogram, numBins * sizeof(float) * grid_size));
  _k_schared_generate_histogram<<<grid_size, block_size>>>(d_logLuminance, min_logLum, 
                                                           lum_range, d_histogram, 
                                                           input_size);

  unsigned int *d_histogram_input = d_histogram;
  while(gridSize > 1)
  {
      checkCudaErrors(cudaMemMalloc(&d_histogram, numBins * sizeof(float) * grid_size));

      _k_schared_reduce_histogram<<<grid_size, block_size>>>(d_histogram_input, d_histogram, 
                                                             input_size, numBins);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      checkCudaErrors(cudaFree(d_histogram_input));
      d_histogram_input = d_histogram;
      input_size = grid_size;
      grid_size = (grid_size + NUM_THREAD - 1) / NUM_THREAD;
  }

  return d_histogram_input;
}

__global__
unsigned int* _k_schared_generate_histogram(const float* const d_logLuminance, 
                                            const float &min_logLum, 
                                            const float &lum_range, 
                                            unsigned int* const d_histogram, 
                                            const size_t input_size)
{
  size_t my_absolute_position = blockDim.y * blockIdx.x + threadIdx.x;
  size_t int thread_id = threadIdx.x;

  extern __shared__ unsigned int *shared_bins[];
  if (my_absolute_position < input_size) {
      unsigned char bin_positon = (d_luminance[my_absolute_position] - 
                                   min_logLum) / lum_range * numBins;
      shared_bins[thread_id * numBins + bin_positon]++;
  }
  else {
    return;
  }

  __syncthreads;

  for (int i = threadDim.x / 2; i > 0; i>>=1)
  {
    if (thread_id < i)
    {
      for (int j = 0; j < numBins; ++j)
      {
        shared_bins[thread_id * numBins + j] += shared_bins[(thread_id + i) * 
                                                numBins + j];
      }
    }
  }

  __syncthreads();

  if (thread_id = 0)
  {
    for (int j = 0; j < numBins; ++j)
    {
      d_histogram[blockIdx.x * numBins + j] += shared_bins[j];
    }
  }

}

__global__
void _k_schared_reduce_histogram(const float* const d_histogram_input, 
                                 float* const d_histogram, 
                                 const size_t input_size, 
                                 const size_t numBins)
{
  size_t my_absolute_position = blockDim.y * blockIdx.x + threadIdx.x;
  int thread_id = threadIdx.x;

  extern __shared__ unsigned int d_shared_histogram[];
  if (my_absolute_position < input_size) {

    for (int i = 0; i < numBins; ++i) {
      d_shared_histogram[thread_id * numBins + i] = d_histogram_input[my_absolute_position * numBins + i];
    }
  }
  else {
    return;
  }

  __syncthreads();

for (int i = blockDim.x ; i > 0; i >>= 1)
{
  if (thread_id < i)
  {
    for (int j = 0; j < numBins; ++j) {
      d_shared_histogram[thread_id * numBins + j] += d_shared_histogram[(thread_id + i) * numBins + j];
    }
  }

  __syncthreads();
}
  
  if (thread_id == 0)
  {
    for (int j = 0; j < numBins; ++j) {
      d_histogram[blockIdx.x * numBins + j] += d_shared_histogram[j];
    }
  }

}

__global__
void _k_perform_scan(const unsigned int* const d_histogram, 
                     unsigned int* const d_cdf, 
                     size_t const numBins) {
  size_t my_absolute_position = blockDim.x + blockIdx.x + threadIdx.x;
  unsigned int thread_id = threadIdx.x;

  if (my_absolute_position >= numBins) {
    return;
  }

  
}