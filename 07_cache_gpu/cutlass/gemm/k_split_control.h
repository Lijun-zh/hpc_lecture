/******************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

/**
 * \file
 * Abstraction for coordinating inter-block k-splitting
 */

#include <stdint.h>

#include "../util/util.h"

namespace cutlass {
namespace gemm {


/******************************************************************************
 * Storage and initialization
 ******************************************************************************/

enum
{
    NumFlagsSplitK = 4096
};


/**
 * Global K-split semaphore flags
 *
 * TODO: use demand-allocated storage to provide copies for concurrent streams
 */
__device__ int d_flags_split_k[NumFlagsSplitK];


/******************************************************************************
 * k_split_control
 ******************************************************************************/

/**
 * \brief Abstraction for coordinating inter-block k-splitting
 */
struct k_split_control
{
    /// Extent of a thread block's partition along the GEMM K-axis
    int split_k;

    /// Whether or not to use a semaphore for inter-block k-splitting.
    bool use_semaphore;

    /// Pointer to semaphore
    int *d_flags;



    //-------------------------------------------------------------------------
    // Device API
    //-------------------------------------------------------------------------

    /**
     * Return the thread block's starting coordinate (k) within the
     * multiplicand matrices
     */
    inline __device__
    int block_begin_item_k()
    {
        return blockIdx.z * split_k;
    }


    /**
     * Return the thread block's ending coordinate (k) within the multiplicand
     * matrices (one-past)
     */
    inline __device__
    int block_end_item_k(int dim_k)
    {
        int next_start_k = block_begin_item_k() + split_k;
        return __NV_STD_MIN(next_start_k, dim_k);
    }


    /**
     * Whether the thread block is a secondary accumulator in an inter-block
     * k-splitting scheme
     */
    inline __device__
    bool is_secondary_accumulator()
    {
        return (blockIdx.z > 0);
    }

    //-------------------------------------------------------------------------
    // Grid launch API
    //-------------------------------------------------------------------------

    /**
     * Constructor
     */
    inline
    k_split_control(
        int     *d_flags,
        int     sm_count,
        int     max_sm_occupancy,
        int     dim_k,
        int     block_tile_items_k,
        dim3    block_dims,
        dim3    &grid_dims)         ///< [in,out]
    :
        d_flags(d_flags),
        split_k(dim_k)
    {}

};


} // namespace gemm
} // namespace cutlass
