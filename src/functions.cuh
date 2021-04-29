#pragma once
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"


#include <stdio.h>

namespace CudaRalaFunctions {
    void cuda_fvr_mean(uint32_t *overlap_begins, uint32_t *overlap_ends,        // Overlap begins and ends
                       uint32_t *pile_begins,    uint32_t *pile_ends,           // Pile begins and ends
                       bool *valid_regions,      float *means);                 // Sets valid regions and means (if valid == true)

    
}