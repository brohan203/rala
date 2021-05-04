// Cuda functions replacing those in piles.cpp and piles.hpp

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <xmmintrin.h>

// CUDA runtime
//#include <cuda_runtime.h>
// Helper functions and utilities to work with CUDA
//#include "include/helper_functions.h"
//#include "include/helper_cuda.h"
//#include "include/device_launch_parameters.h"

#include <stdio.h>

// Wrapper file
#include "functions.cuh"

// Implementation of upper_bound using binary search. Basically copied from C++ Standard Library
__device__ uint32_t cuda_upper_bound(uint32_t *overlaps, uint32_t first, uint32_t last, const uint32_t location) {
    uint32_t it, count, step;
    count = last - first;

    while(count > 0) {
        it = first;
        step = count / 2;
        it = it + step;
        if(!(location < overlaps[it])) {
            first = ++it;
            count -= step + 1;
        }
        else {
            count = step;
        }
    }
    return first - 1;
}


__device__ uint32_t cuda_find_histo_height(uint32_t location, uint32_t *overlap_begins, uint32_t *overlap_ends) {

    int arr_length = sizeof(overlap_begins) / sizeof(overlap_begins[0]);

    // printf("New FHH used %d iterations\n", i);
    return ( cuda_upper_bound(overlap_begins, 0, arr_length-1, location) - 
             cuda_upper_bound(overlap_ends,   0, arr_length-1, location)  );
}

// Combination of find_valid_regions, find_mean (replacement for find_median), find_chimeric_hills, and find_chimeric_pits
__global__ void cuda_fvr_mean(uint32_t *overlap_begins,             // Overlap begins and ends
                              uint32_t *overlap_ends, 
                              uint32_t *pile_begins,                // Specific begins and ends of the piles
                              uint32_t *pile_ends, 
                              bool *valid_regions, float *means)    // Bool vector for resetting piles
{

    // Figure out where we are. We will be using the "gid" position within all arrays
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t new_begin = 0, new_end = 0, current_begin = 0;
    bool found_begin = false;
    for (uint32_t i = pile_begins[gid]; i < pile_ends[gid]; ++i) {
        // Using height function here. Saving as variable because we use it twice
        uint32_t this_height = cuda_find_histo_height(i, overlap_begins, overlap_ends);
        if (!found_begin && this_height >= 4) {
            current_begin = i;
            found_begin = true;
        } else if (found_begin && this_height < 4) {
            if (i - current_begin > new_end - new_begin) {
                new_begin = current_begin;
                new_end = i;
            }
            found_begin = false;
        }
    }
    if (found_begin) {
        if (pile_ends[gid] - current_begin > new_end - new_begin) {
            new_begin = current_begin;
            new_end = pile_ends[gid];
        }
    }

    // If not a valid region, return (and skip mean)
    if (pile_ends[gid] - pile_begins[gid] < 1260) {
        valid_regions[gid] = false;
        means[gid] = 0.;
        return;
    }

    // If valid region, continue with mean + chimeric_hills + chimeric_pits
    else {
        pile_begins[gid] = new_begin;
        pile_ends[gid] = new_end;
        valid_regions[gid] =  true;

        // Find_mean (replacement for find_median)
        int mean = 0;

        for(int i = pile_begins[gid]; i < pile_ends[gid]; ++i){
            mean = mean + cuda_find_histo_height(i, overlap_begins, overlap_ends);
        }

        means[gid] = mean;
    }
}


// NOTE: variables beginning with "h_" denote those on host (cpu), variables beginning with "d_" denotes those on device (gpu)
namespace CudaRalaFunctions {
    void fvr_mean(uint32_t *h_overlap_begins, uint32_t *h_overlap_ends,       // Overlap begins and ends
                  uint32_t *h_pile_begins,    uint32_t *h_pile_ends,          // Pile begins and ends
                  bool *h_valid_regions,      float *h_means) {               // Empty bool and float arrays, respectively

            // Device initiation
            int dev = findCudaDevice();

            // We will use "status" to ensure everything is going smoothly
            cudaError_t status;

            // CUDA events that we'll use for timing:
            cudaEvent_t function_start, gpu_start, stop;
            status = cudaEventCreate( &function_start );
            checkCudaErrors( status );
            status = cudaEventCreate( &gpu_start );
            checkCudaErrors( status );
            status = cudaEventCreate( &stop );
            checkCudaErrors( status );

            // record the start event:
            status = cudaEventRecord( function_start, NULL );
            checkCudaErrors( status );

            // Allocate device memory. We have 5 mallocs here.
            uint32_t *d_overlap_begins, *d_overlap_ends;
            uint32_t *d_pile_begins, *d_pile_ends;
            bool *d_valid_regions;
            float *d_means;

            uint32_t num_overlaps = sizeof(overlap_begins) / sizeof(overlap_begins[0]);
            uint32_t num_piles    = sizeof(pile_begins)    / sizeof(pile_begins[0]);

            // Allocate memory on GPU
            // Overlap begins/ends
            status = cudaMalloc( (void **)(&d_overlap_begins), num_overlaps*sizeof(uint32_t) );
            checkCudaErrors( status );
            status = cudaMalloc( (void **)(&d_overlap_ends),   num_overlaps*sizeof(uint32_t) );
            checkCudaErrors( status );
            // Allocate pile_begins and pile_ends. Will be edited for valid regions within kernel
            status = cudaMalloc( (void **)(&d_pile_begins),    num_piles*sizeof(uint32_t) );
            checkCudaErrors( status );
            status = cudaMalloc( (void **)(&d_pile_ends),      num_piles*sizeof(uint32_t) );
            checkCudaErrors( status );
            // Allocate valid_regions (bool*) and means (float*).
            // The former will determine which regions are valid, the latter returns calculated means for regions that are valid
            status = cudaMalloc( (void **)(&d_valid_regions),  num_piles*sizeof(bool) );
            checkCudaErrors( status );
            status = cudaMalloc( (void **)(&d_means),          num_piles*sizeof(float) );
            checkCudaErrors( status );

            // Copy host memory to the device (ram to GPU memory). We have four copies here
            // Copy overlap begins/ends
            status = cudaMemcpy( d_overlap_begins, h_overlap_begins, num_overlaps*sizeof(uint32_t), cudaMemcpyHostToDevice );
            checkCudaErrors( status );
            status = cudaMemcpy( d_overlap_ends,   h_overlap_ends,   num_overlaps*sizeof(uint32_t), cudaMemcpyHostToDevice );
            checkCudaErrors( status );
            // Copy pile begins/ends
            status = cudaMemcpy( d_pile_begins,    h_pile_begins,    num_piles*sizeof(uint32_t),    cudaMemcpyHostToDevice );
            checkCudaErrors( status );
            status = cudaMemcpy( d_pile_ends,      h_pile_ends,      num_piles*sizeof(uint32_t),    cudaMemcpyHostToDevice );
            checkCudaErrors( status );

            // Establish block size and calculate how many blocks we need to use
            int blocksize = 32;
            int numblocks = ceil(num_piles / blocksize);

            // setup the execution parameters:
            dim3 threads(blocksize, 1, 1 );
            dim3 grid(numblocks, 1, 1 );

            // create and start timer
            cudaDeviceSynchronize( );

            // record the start event:
            status = cudaEventRecord( gpu_start, NULL );
            checkCudaErrors( status );

            // execute the kernel:
            cuda_fvr_mean<<< grid, threads >>>(d_overlap_begins, d_overlap_ends,
                                               d_pile_begins, d_pile_ends,         // Vectors to edit
                                               d_valid_regions, d_means) );        // Vectors to fill

            // record the stop event:
            status = cudaEventRecord( stop, NULL );
            checkCudaErrors( status );
            // wait for the stop event to complete:
            status = cudaEventSynchronize( stop );
            checkCudaErrors( status );
            
            double msecGPU = 0.0f;
            status = cudaEventElapsedTime( &msecGPU,   gpu_start,      stop );
            double secondsGPU   = 0.001 * (double)msecGPU;
            double pilesPerSecondGPU   = (float)num_piles / secondsGPU;

            float msecTotal = 0.0f;
            status = cudaEventElapsedTime( &msecTotal, function_start, stop );
            checkCudaErrors( status );

            // compute and print the performance
            double secondsTotal = 0.001 * (double)msecTotal;
            double pilesPerSecondTotal = (float)num_piles / secondsTotal;

            // Copy results from the device to the host:
            // New pile begins and ends
            status = cudaMemcpy( &h_pile_begins,   d_pile_begins,   num_piles*sizeof(uint32_t), cudaMemcpyDeviceToHost );
            checkCudaErrors( status );
            status = cudaMemcpy( &h_pile_ends,     d_pile_ends,     num_piles*sizeof(uint32_t), cudaMemcpyDeviceToHost );
            checkCudaErrors( status );
            // Bool array of valid regions
            status = cudaMemcpy( &h_valid_regions, d_valid_regions, num_piles*sizeof(bool),     cudaMemcpyDeviceToHost );
            checkCudaErrors( status );
            // Float array of means
            status = cudaMemcpy( &h_means,         d_valid_regions, num_piles*sizeof(bool),     cudaMemcpyDeviceToHost );
            checkCudaErrors( status );
            cudaDeviceSynchronize( );
    }
}