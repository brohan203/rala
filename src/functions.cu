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
#include <cuda_runtime.h>
// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"

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
             cuda_upper_bound(overlap_ends,   0, arr_length-1, location    ) );
}

// Combination of find_valid_regions, find_mean (replacement for find_median), find_chimeric_hills, and find_chimeric_pits
// ***** Currently only has find valid regions and find mean
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