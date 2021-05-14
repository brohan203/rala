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


__device__ uint32_t cuda_find_histo_height(uint32_t location, uint32_t &pile_begins, uint32_t *overlap_begins, uint32_t *overlap_ends) {
    // Figure out where we are. We will be using the "gid" position within all arrays
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    thisLocation = pile_begins[gid] + location;

    int arr_length = sizeof(overlap_begins) / sizeof(overlap_begins[0]);

    // printf("New FHH used %d iterations\n", i);
    return ( cuda_upper_bound(overlap_begins, 0, arr_length-1, thisLocation) - 
             cuda_upper_bound(overlap_ends,   0, arr_length-1, thisLocation)  );
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
        uint32_t this_height = cuda_find_histo_height(i, pile_begins, overlap_begins, overlap_ends);
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
            mean = mean + cuda_find_histo_height(i, pile_begins, overlap_begins, overlap_ends);
        }

        means[gid] = mean;
    }
}



__device__ void cuda_find_slopes(double q, uint32_t slope_region_counter,
                                 uint32_t &pile_begins, uint32_t &pile_ends,
                                 uint32_t &slope_region_begins, uint32_t &slope_region_ends, 
                                 uint32_t &overlap_begins, uint32_t &overlap_ends) {

    // Figure out where we are. We will be using the "gid" position within all arrays
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
 
    int32_t k = 847;
    int32_t read_length = pile_ends[gid] - pile_begins[gid];

    // Turned subpiles into arrays
    // Down corresponds to left subpile
    uint32_t l_subpile[k];
    memset( l_subpile, 0, k*sizeof(uint32_t));                                    // To delete
    uint32_t first_down = 0, last_down = 0;
    bool found_down = false;

    // Up corresponds to right subpile
    uint32_t r_subpile[k];
    memset( r_subpile, 0, k*sizeof(uint32_t));
    uint32_t first_up = 0, last_up = 0;
    bool found_up = false;

    // find slope regions
    for (int32_t i = 0; i < k; ++i) {
        r_subpile[i] = cuda_find_histo_height(i, pile_begins, overlap_begins, overlap_ends);
    }

    // For loop to build initial slope_regions
    for (int32_t i = 0; i < read_length; ++i) {

        // Last position in the array
        uint32_t subpile_end   = i % k;
        // First position in the array
        uint32_t subpile_begin = (i+1) % k;

        if (i > 0) {
            l_subpile[subpile_end] = cuda_find_histo_height(i - 1, pile_begins, overlap_begins, overlap_ends);
        }

        if (i < read_length - k) {
            r_subpile[subpile_end] = cuda_find_histo_height(i + k, pile_begins, overlap_begins, overlap_ends);
        }

        uint32_t left_max  = 0;
        uint32_t right_max = 0;
        for(int x = 0; x < k; ++x) {
            if(l_subpile[x] > left_max) {
                left_max = l_subpile[x];
            }
            if(r_subpile[x] > right_max) {
                right_max = r_subpile[x];
            }
        }

        int32_t current_value = cuda_find_histo_height(i, pile_begins, overlap_begins, overlap_ends) * q;
        // Set last down if above a certain threshold
        if (i != 0 && left_max > current_value) {
            if (found_down) {
                if (i - last_down > 1) {
                    slope_region_begins[slope_region_counter] = first_down << 1 | 0;
                    slope_region_ends[slope_region_counter]   = last_down;
                    ++slope_region_counter;
                    first_down = i;
                }
            } else {
                found_down = true;
                first_down = i;
            }
            last_down = i;
        }
        if (i != (read_length - 1) && right_max > current_value) {
            if (found_up) {
                if (i - last_up > 1) {
                    slope_region_begins[slope_region_counter] = first_up << 1 | 1;
                    slope_region_ends[slope_region_counter]   = last_up;
                    ++slope_region_counter;
                    first_up = i;
                }
            } else {
                found_up = true;
                first_up = i;
            }
            last_up = i;
        }
    }
    if (found_down) {
        slope_region_begins[slope_region_counter] = first_down << 1 | 0;
        slope_region_ends[slope_region_counter]   = last_down;
        ++slope_region_counter;
    }
    if (found_up) {
        slope_region_begins[slope_region_counter] = first_up << 1 | 1;
        slope_region_ends[slope_region_counter]   = last_up;
        ++slope_region_counter;
    }

    if (slope_region_counter == 0) {
        return;
    }
    
    while (true) {
        // Implementation of insertion sort for our slope_region_begins/ends arrays
        // Sorts based on slope_region_begins
        int i, key_begins, key_ends, j;
        for (i = 1; i < slope_region_counter; i++)
        {
            key_begins = slope_region_begins[i];
            key_ends = slope_region_ends[i];
            j = i - 1;
    
            /* Move elements of arr[0..i-1], that are
            greater than key, to one position ahead
            of their current position */
            while (j >= 0 && slope_region_begins[j] > key_begins)
            {
                slope_region_begins[j + 1] = slope_region_begins[j];
                slope_region_ends[j + 1] = slope_region_ends[j];
                j = j - 1;
            }
            slope_region_begins[j + 1] = key_begins;
            slope_region_ends[j + 1] = key_ends;
        }

        // for(int x = 0; x < slope_region_counter; ++x) {
        //     // if(slope_regions[x].first != slope_region_begins[x]) {
        //     //     printf("begins discrepancy after sort: theirs %d, mine %d\n", slope_regions[x].first, slope_region_begins[x]);
        //     // }
        //     // if(slope_regions[x].second != slope_region_ends[x]) {
        //     //     printf("ends discrepancy after sort: theirs %d, mine %d\n", slope_regions[x].second, slope_region_ends[x]);
        //     // }
        //     printf("mine %d-%d, theirs %d-%d\n", slope_region_begins[x], slope_region_ends[x], slope_regions[x].first, slope_regions[x].second);
        // }

        bool is_changed = false;
        for (uint32_t i = 0; i < slope_region_counter - 1; ++i) {
            if (slope_region_ends[i] < (slope_region_begins[i + 1] >> 1)) {
                continue;
            }

            if (slope_region_begins[i] & 1) {
                found_up = false;

                // Set subpile begin to the begin
                uint32_t subpile_begin = slope_region_begins[i] >> 1;
                // Set subpile end to MIN(ends[i], ends[i+1])
                uint32_t subpile_end = slope_region_ends[i];
                if(subpile_end > slope_region_ends[i+1]) {
                    subpile_end = slope_region_ends[i + 1]);
                }
                uint32_t region_width = subpile_end - subpile_begin + 1;

                uint32_t r_subpileregion [region_width + 2];
                memset(r_subpileregion, 0, (region_width + 2)*sizeof(uint32_t));

                for (uint32_t j = subpile_begin; j < subpile_end + 1; ++j) {
                    r_subpileregion[j % (region_width)] = cuda_find_histo_height(j, pile_begins, overlap_begins, overlap_ends);
                }

                for (uint32_t j = subpile_begin; j < subpile_end; ++j) {

                    r_subpileregion[j % region_width] = 0;
                    
                    // Initialize and find max of r_subpileregion
                    uint32_t right_max = 0;
                    for (uint32_t max = 0; max < (subpile_end - subpile_begin + 1); ++max) {
                        if(right_max < r_subpileregion[max]) {
                            right_max = r_subpileregion[max];
                        }
                    }

                    if (cuda_find_histo_height(j, pile_begins, overlap_begins, overlap_ends) * q < right_max) {
                        if (found_up) {
                            if (j - last_up > 1) {
                                slope_region_begins[slope_region_counter] = first_up << 1 | 1;
                                slope_region_ends[slope_region_counter] = last_up;
                                ++slope_region_counter;
                                first_up = j;
                            }
                        } else {
                            found_up = true;
                            first_up = j;
                        }
                        last_up = j;
                    }
                }
                if (found_up) {
                    slope_region_begins[slope_region_counter] = first_up << 1 | 1;
                    slope_region_ends[slope_region_counter] = last_up;
                    ++slope_region_counter;
                }

                slope_region_begins[i] = subpile_end << 1 | 1;

            } else {
                //printf("else");
                if (slope_region_ends[i] == (slope_region_begins[i + 1] >> 1)) {
                    continue;
                }

                found_down = false;

                // Set subpile begin to MAX(begins[i], begins[i+1])
                uint32_t subpile_begin = slope_region_begins[i] >> 1;
                if(subpile_begin < slope_region_begins[i+1] >> 1) {
                    subpile_begin = slope_region_begins[i + 1] >> 1;
                }
                // Set subpile end to current position end
                uint32_t subpile_end = slope_region_ends[i];
                uint32_t region_width = subpile_end - subpile_begin + 1;

                uint32_t left_max = 0;

                for (uint32_t j = subpile_begin; j < subpile_end + 1; ++j) {
                    if ((j != subpile_begin) && data_[j] * q < left_max) {
                        if (found_down) {
                            if (j - last_down > 1) {
                                slope_region_begins[slope_region_counter] = first_down << 1 | 0;
                                slope_region_ends[slope_region_counter] = last_down;
                                ++slope_region_counter;
                                first_down = j;
                            }
                        } else {
                            found_down = true;
                            first_down = j;
                        }
                        last_down = j;
                    }
                    if(left_max < cuda_find_histo_height(j, pile_begins, overlap_begins, overlap_ends)) {
                        left_max = cuda_find_histo_height(j, pile_begins, overlap_begins, overlap_ends);
                    }
                }
                if (found_down) {
                    slope_region_begins[slope_region_counter] = first_down << 1 | 0;
                    slope_region_ends[slope_region_counter] = last_down;
                    ++slope_region_counter;
                }
                slope_region_ends[i] = subpile_begin;
            }

            is_changed = true;
            break;
        }

        if (!is_changed) {
            break;
        }
    }

    // narrow slope regions
    for (uint32_t i = 0; i < slope_region_counter - 1; ++i) {
        if ((slope_region_begins[i] & 1) && !(slope_region_begins[i + 1] & 1)) {

            uint32_t subpile_begin = slope_region_ends[i];
            uint32_t subpile_end = slope_region_ends[i + 1] >> 1;

            if (subpile_end - subpile_begin > static_cast<uint32_t>(k)) {
                continue;
            }

            uint16_t max_subpile_coverage = 0;
            for (uint32_t j = subpile_begin + 1; j < subpile_end; ++j) {
                max_subpile_coverage = std::max(max_subpile_coverage, (uint16_t)cuda_find_histo_height(j, pile_begins, overlap_begins, overlap_ends));
            }

            uint32_t last_valid_point = slope_region_begins[i] >> 1;
            for (uint32_t j = slope_region_begins[i] >> 1; j <= subpile_begin; ++j) {
                if (max_subpile_coverage > cuda_find_histo_height(j, pile_begins, overlap_begins, overlap_ends) * q) {
                    last_valid_point = j;
                }
            }

            uint32_t first_valid_point = slope_region_ends[i + 1];
            for (uint32_t j = subpile_end; j <= slope_region_ends[i + 1]; ++j) {
                if (max_subpile_coverage > cuda_find_histo_height(j, pile_begins, overlap_begins, overlap_ends) * q) {
                    first_valid_point = j;
                    break;
                }
            }

            slope_region_ends[i] = last_valid_point;
            slope_region_begins[i + 1] = first_valid_point << 1 | 0;
        }
    }
}

__global__ void find_chimeric_pits(uint32_t &slope_regions, uint32_t &pile_begins, uint32_t &pile_ends, 
                                   uint32_t &overlap_begins, uint32_t &overlap_ends) {
    if(id_ % 100 == 0) {
        printf("find_pits on pile %d\n", id_);
    }
    uint32_t slope_region_begins [847];
    memet(slope_region_begins, 0, sizeof(uint32_t)*847);
    uint32_t slope_region_ends [847];
    memet(slope_region_ends, 0, sizeof(uint32_t)*847);
    uint32_t slope_region_counter;

    // Fills slope_region_begins/ends with proper slope info
    // Fills counter with length of array
    cuda_find_slopes(1.82, slope_region_counter, 
                     pile_begins, pile_ends, 
                     slope_region_begins, slope_region_ends, 
                     overlap_begins, overlap_ends);

    if (slope_region_counter == 0) {
        return;
    }

    for (uint32_t i = 0; i < region_length - 1; ++i) {
        if (!(slope_region_begins[i] & 1) && (slope_region_begins[i + 1] & 1)) {
            chimeric_pits_.emplace_back(slope_regions[i].first >> 1, slope_regions[i + 1].second);  // FIX THIS
        }
    }
    intervalMerge(chimeric_pits_);
}

__global__ void find_chimeric_hills(uint32_t &pile_begins, uint32_t &pile_ends, 
                                    uint32_t &overlap_begins, uint32_t &overlap_ends) {
    // Figure out where we are. We will be using the "gid" position within all arrays
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(id_ % 100 == 0) {
        printf("find_hills on pile %d\n", id_);
    }
    uint32_t slope_region_begins [847];
    memet(slope_region_begins, 0, sizeof(uint32_t)*847);
    uint32_t slope_region_ends [847];
    memet(slope_region_ends, 0, sizeof(uint32_t)*847);
    uint32_t slope_region_counter;

    cuda_find_slopes(1.3, slope_region_counter, 
                     pile_begins, pile_ends, 
                     slope_region_begins, slope_region_ends, 
                     overlap_begins, overlap_ends);

    if (slope_region_counter == 0) {
        return;
    }

    auto is_chimeric_hill = [&](
        const std::pair<uint32_t, uint32_t>& begin,
        const std::pair<uint32_t, uint32_t>& end) -> bool {

        // If it's at the beginning or the end, return false
        if ((begin.first >> 1) < 0.05 * (this->end_ - this->begin_) + this->begin_ ||
            end.second > 0.95 * (this->end_ - this->begin_) + this->begin_ ||
            (end.first >> 1) - begin.second > 840) {
            return false;
        }

        // 
        uint32_t peak_value = 1.3 * std::max(cuda_find_histo_height(begin.second, overlap_begins, overlap_ends),
                                             cuda_find_histo_height(end.first >> 1, overlap_begins, overlap_ends));

        for (uint32_t i = begin.second + 1; i < (end.first >> 1); ++i) {
            if (cuda_find_histo_height(i, pile_begins, overlap_begins, overlap_ends) > peak_value) {
                return true;
            }
        }
        return false;
    };

    uint32_t fuzz = 420;
    for (uint32_t i = 0; i < slope_region_counter - 1; ++i) {
        if (!(slope_region_beginss[i] & 1)) {
            continue;
        }

        for (uint32_t j = i + 1; j < slope_region_counter; ++j) {
            if (slope_region_begins[j] & 1) {
                continue;
            }
            
            if (is_chimeric_hill(slope_regions[i], slope_regions[j])) {
                uint32_t begin = (slope_regions[i].first >> 1) - this->begin_ > fuzz ?
                    (slope_regions[i].first >> 1) - fuzz : this->begin_;
                uint32_t end = this->end_ - slope_regions[j].second > fuzz ?
                    slope_regions[j].second + fuzz : this->end_;
                chimeric_hills_.emplace_back(begin, end);
            }
        }
    }
    intervalMerge(chimeric_hills_);

    chimeric_hill_coverage_.resize(chimeric_hills_.size(), 0);
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