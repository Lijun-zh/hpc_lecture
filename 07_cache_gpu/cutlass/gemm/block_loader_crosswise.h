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
 * Tile-loading abstraction for thread blocks
 */

#include "../util/util.h"

namespace cutlass {
namespace gemm {


/******************************************************************************
 * block_loader (CrosswiseCopy specialization)
 ******************************************************************************/

/**
 * \brief A three-phase data loading abstraction (prefetch, commit, and
 * advance) for iterating over ranges of block-wide matrix tiles.
 * (CrosswiseCopy specialization)
 *
 * Each iteration sequence produces a KxL (height-by-width) block-wide tile of
 * value_t in shared memory.  The layout of the shared block-wide tile is
 * a row-major (L-major) tiling of dp_vector_t items, which are themselves
 * column-major (K-major) vectors of value_t.  Its dimensions are:
 *    K = BlockDpVectorsK * (sizeof(dp_vector_t) / sizeof(value_t)
 *    L = BlockDpVectorsL
 *
 * The data is copied from a corresponding tile of global matrix data whose
 * layout of value_t is K-major.  This constitutes a CrosswiseCopy between
 * the K-major global tile and the L-major shared tile.
 *
 * NB: The orientation of dp_vector_t components in shared memory is congruous
 * with the global matrix data, so we can use dp_vector_t as the minimum
 * granularity of data transfer without any intermediate {dis|re}assembly
 * of its value_t components.  However, the global and shared memory layouts
 * of dp_vector_t items are cross-wise with respect to each other, so any
 * further LDG-vectorization of dp_vector_t data requires intermediate
 * disassembly into dp_vector_t components to be stored individually into
 * the shared tile.
 *
 * NB: Consecutive threads within a block are mapped in K-major
 * fashion down a first set of LDG-vectors of dp_vector_t within their global
 * tile. Successive sets of LDG-vectors are then strip-mined as necessary
 * across the L-axis.  These discontiguous LDG-vectors comprise the thread's
 * "slice" of the block-wide tile.
 */
template <
    int BlockThreads,           ///< Number of threads in each thread block (blockDim.x)
    int BlockDpVectorsK,        ///< Extent of block-wide tile in dp_vector_t along the K-axis (height)
    int BlockDpVectorsL,        ///< Extent of block-wide tile in dp_vector_t along the L-axis (width)
    typename value_t,           ///< Input matrix value type
    int LeadingDimAlignBytes,   ///< Byte alignment of input matrix leading dimension
    bool AllowRaggedTiles,      ///< Whether the input matrix's dimensions need not be an even-multiple of the block-wide tile dimensions
    typename dp_vector_t>       ///< Dot-product vector type along the K-axis
struct block_loader<
    BlockThreads,
    BlockDpVectorsK,
    BlockDpVectorsL,
    value_t,
    LeadingDimAlignBytes,
    AllowRaggedTiles,
    dp_vector_t,
    load_algorithm::CrosswiseCopy>  ///< Algorithm for loading a shared tile of KxL matrix data (CrosswiseCopy specialization)
{
    //-------------------------------------------------------------------------
    // Constants and types
    //-------------------------------------------------------------------------

    enum
    {
        /// Number of value_t in a dp_vector_t
        DpVectorItems = divide_assert<sizeof(dp_vector_t), sizeof(value_t)>::value,

        /// Number of dp_vector_t in a block-wide tile
        BlockDpVectors = BlockDpVectorsK * BlockDpVectorsL,

        /// Number of dp_vector_t in a thread-tile
        ThreadDpVectors = divide_assert<BlockDpVectors, BlockThreads>::value,
    };

    /// Data movement type, coarsened by LeadingDimAlignBytes, capped by the
    /// smaller of either ThreadDpVectors or BlockDpVectorsK
    typedef io_vector<
            dp_vector_t,
            __NV_STD_MIN(ThreadDpVectors, BlockDpVectorsK),
            LeadingDimAlignBytes>
        ldg_vector_t;

    enum
    {
        /// Number of dp_vector_t per ldg_vector_t
        LdgVectorDpVectors = ldg_vector_t::VectorItems,

        /// Number of value_t per ldg_vector_t
        LdgVectorItems = LdgVectorDpVectors * DpVectorItems,



        /// Total number of ldg_vector_t within each block-wide tile
        BlockLdgVectors = divide_assert<BlockDpVectors, LdgVectorDpVectors>::value,

        /// Extent of the block-wide tile in ldg_vector_t along K-axis
        BlockLdgVectorsK = divide_assert<BlockDpVectorsK, LdgVectorDpVectors>::value,

        /// Extent of the block-wide tile in ldg_vector_t along L-axis
        BlockLdgVectorsL = BlockDpVectorsL,



        /// Number of ldg_vector_t within each thread-tile
        ThreadLdgVectors = divide_assert<BlockLdgVectors, BlockThreads>::value,

        /// Extent of the thread tile in ldg_vector_t along K-axis
        ThreadLdgVectorsK = __NV_STD_MAX(1, (BlockLdgVectorsK / BlockThreads)),

        /// Extent of the thread tile in ldg_vector_t along L-axis
        ThreadLdgVectorsL = divide_assert<ThreadLdgVectors, ThreadLdgVectorsK>::value,



        /// Number of ldg_vector_t within each stripmine-tile
        StripmineLdgVectors = BlockThreads,

        /// Extent of the stripmine tile in ldg_vector_t along K-axis
        StripmineLdgVectorsK = __NV_STD_MIN(BlockLdgVectorsK, StripmineLdgVectors),

        /// Extent of the stripmine tile in ldg_vector_t along L-axis
        StripmineLdgVectorsL = divide_assert<StripmineLdgVectors, StripmineLdgVectorsK>::value,



        /// Alignment in dp_vector_t along L needed for committing prefetch
        AlignmentDpVectorsL = 1,
    };

    /// Predicate bit vector
    typedef uint64_t predicate_mask_t;


    //-------------------------------------------------------------------------
    // Assert assumptions
    //-------------------------------------------------------------------------

    static_assert(
        (ThreadLdgVectors <= sizeof(predicate_mask_t) * 8),
        "Predicate mask type does not contain enough bits for encoding load predicates");


    //-------------------------------------------------------------------------
    // Members
    //-------------------------------------------------------------------------

    /// Input pointer to matrix in ldg_vector_t
    ldg_vector_t *d_matrix_ldgvecs;

    /// Extent of the input matrix in ldg_vector_t along the L-axis
    int matrix_ldgvecs_l;

    /// Thread block's ending ldg_vector_t coordinate (k) within the input matrix (one-past)
    int block_end_ldgvec_k;

    /// Predicate bits for guarding ldg_vector_t loads within "whole-k" block-wide tiles
    predicate_mask_t guard;

    /// Predicate bits for guarding ldg_vector_t loads within the final block-wide "residue" tile
    predicate_mask_t residue_guard;

    /// Iteration span in "whole-k" block-wide tiles
    int wholek_tiles_remaining;

    /// Distance in ldg_vector_t within pitched-linear memory between successive coordinates along the K-axis
    int matrix_ldgvec_stride_k;

    /// Distance in ldg_vector_t within pitched-linear memory between successive coordinates along the L-axis
    int matrix_ldgvec_stride_l;

    /// ldg_vector_t coordinates (l, k) of thread-tile within the block-wide tile
    int2 block_thread_ldgvec_coords;

    /// Thread-wide tile of prefetch data
    ldg_vector_t thread_tile[ThreadLdgVectorsK][ThreadLdgVectorsL];


    //-------------------------------------------------------------------------
    // Constructor API
    //-------------------------------------------------------------------------

    /// Constructor
    inline __device__
    block_loader(
        value_t *d_matrix_items,        ///< Input pointer to matrix in value_t
        int matrix_items_l,             ///< Extent of the input matrix in value_t along the L-axis
        int matrix_items_stride_k,      ///< Distance in value_t within pitched-linear memory between successive coordinates along the K-axis
        int matrix_items_stride_l,      ///< Distance in value_t within pitched-linear memory between successive coordinates along the L-axis
        int2 matrix_block_item_coords,  ///< value_t coordinates (l, k) of first block-wide tile within the input matrix
        int block_end_item_k)           ///< Thread block's ending coordinate (k) within the input matrix (one-past)
    :
        block_end_ldgvec_k(block_end_item_k),
        guard(0),
        residue_guard(0)
    {
        matrix_ldgvecs_l = matrix_items_l;
        matrix_ldgvec_stride_k = matrix_items_stride_k;
        matrix_ldgvec_stride_l = (matrix_items_stride_l / LdgVectorItems);

        // ldg_vector_t coordinates (l, k) of thread-tile within the block-wide tile
        block_thread_ldgvec_coords = make_int2(
            (threadIdx.x / BlockLdgVectorsK),                // l-coordinate
            (threadIdx.x % BlockLdgVectorsK));               // k-coordinate

        // ldg_vector_t coordinates (l, k) of first block-wide tile within the input matrix
        int2 matrix_block_ldgvec_coords = make_int2(
            matrix_block_item_coords.x,                     // l-coordinate
            matrix_block_item_coords.y / LdgVectorItems);    // k-coordinate

        // Iteration span in ldg_vector_t
        int span_ldgvec_k = (block_end_item_k - matrix_block_item_coords.y) / LdgVectorItems;



        // ldg_vector_t coordinates (l, k) of first thread-tile tile within the input matrix
        int2 matrix_thread_ldgvec_coords = make_int2(
            block_thread_ldgvec_coords.x + matrix_block_ldgvec_coords.x,
            block_thread_ldgvec_coords.y + matrix_block_ldgvec_coords.y);

        // Iteration range in "whole-k" block-wide tiles
        wholek_tiles_remaining = span_ldgvec_k / BlockLdgVectorsK;

        // Update the input pointer to be matrix_thread_ldgvec_coords
        this->d_matrix_ldgvecs =
            reinterpret_cast<ldg_vector_t*>(d_matrix_items) +
            (matrix_thread_ldgvec_coords.y * matrix_ldgvec_stride_k) +
            (matrix_thread_ldgvec_coords.x * matrix_ldgvec_stride_l);
    }


    //-------------------------------------------------------------------------
    // Loader API
    //-------------------------------------------------------------------------

    /**
     * Request the current block-wide tile
     */
    inline __device__
    void request()
    {
        // Outer thread-tile ldg_vector_t iteration (K-axis)
        #pragma unroll
        for (int thread_ldgvec_k = 0; thread_ldgvec_k < ThreadLdgVectorsK; ++thread_ldgvec_k)
        {
            // Inner thread-tile ldg_vector_t iteration (L-axis)
            #pragma unroll
            for (int thread_ldgvec_l = 0; thread_ldgvec_l < ThreadLdgVectorsL; ++thread_ldgvec_l)
            {
                // Linear index of ldg_vector_t load
                int ldgvec_idx = (thread_ldgvec_k * ThreadLdgVectorsL) + thread_ldgvec_l;

                // Unpack predicate guard
                predicate_mask_t valid = ((guard >> ldgvec_idx) & 1);

                if (!AllowRaggedTiles || valid)
                {
                    // Perform load
                    thread_tile[thread_ldgvec_k][thread_ldgvec_l].load(
                        d_matrix_ldgvecs +
                        (thread_ldgvec_k * StripmineLdgVectorsK * matrix_ldgvec_stride_k) +
                        (thread_ldgvec_l * StripmineLdgVectorsL * matrix_ldgvec_stride_l));
                }
                else
                {
                    // Zero-initialize
                    #pragma unroll
                    for (int dpvec = 0; dpvec < LdgVectorDpVectors; ++dpvec)
                        thread_tile[thread_ldgvec_k][thread_ldgvec_l].buff[dpvec] = 0;
                }
            }
        }
    }


    /**
     * Advance the loader to the next block-wide tile in the K-axis
     */
    inline __device__
    void next()
    {
        d_matrix_ldgvecs += (matrix_ldgvec_stride_k * BlockLdgVectorsK);
    }


    /**
     * Commit the previously-requested block-wide tile to shared memory
     *
     * NB: To facilitate padding for avoiding shared memory bank conflicts, we
     * allow the row stride SmemDpVectorsL to be arbitrarily bigger than the
     * tile width BlockDpVectorsL.
     */
    template <int SmemDpVectorsL>
    inline __device__
    void commit(
        dp_vector_t (&scratch_tile)[BlockDpVectorsK][SmemDpVectorsL])
    {
        static_assert(SmemDpVectorsL >= BlockDpVectorsL, "Row stride must be >= tile width.");

        // Outer thread-tile ldg_vector_t iteration (K-axis)
        #pragma unroll
        for (int thread_ldgvec_k = 0; thread_ldgvec_k < ThreadLdgVectorsK; ++thread_ldgvec_k)
        {
            int block_ldgvec_k = block_thread_ldgvec_coords.y + (thread_ldgvec_k * StripmineLdgVectorsK);

            // Inner thread-tile ldg_vector_t iteration (L-axis)
            #pragma unroll
            for (int thread_ldgvec_l = 0; thread_ldgvec_l < ThreadLdgVectorsL; ++thread_ldgvec_l)
            {
                int block_ldgvec_l = block_thread_ldgvec_coords.x + (thread_ldgvec_l * StripmineLdgVectorsL);

                // Write column of dp_vector_t
                #pragma unroll
                for (int dpvec = 0; dpvec < LdgVectorDpVectors; ++dpvec)
                {
                    scratch_tile[(block_ldgvec_k * LdgVectorDpVectors) + dpvec][block_ldgvec_l] =
                        thread_tile[thread_ldgvec_k][thread_ldgvec_l].buff[dpvec];
                }
            }
        }
    }
};


} // namespace gemm
} // namespace cutlass
