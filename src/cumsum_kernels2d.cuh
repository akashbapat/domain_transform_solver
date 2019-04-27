#ifndef CUMSUM_KERNELS2D_H_
#define CUMSUM_KERNELS2D_H_

#include <cub.cuh>

#include <cudaArray2D.h>

namespace domain_transform {
// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
struct BlockPrefixCallbackOp {
  // Running prefix.
  float running_total;
  // Constructor.
  __device__ BlockPrefixCallbackOp(float running_total)
      : running_total(running_total) {}
  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide
  // scan.
  __device__ float operator()(float block_aggregate) {
    float old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

// Integrates in parallel along the horizontal direction of a 2D image.
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void SegmentedScanParallel2DHorizontal(
    const ImageDim image_dim, const cua::CudaArray2D<float> differential,
    cua::CudaArray2D<float> integral) {
  // Specialize BlockLoad type for our thread block (uses warp-striped loads for
  // coalescing, then transposes in shared memory to a blocked arrangement).
  typedef cub::BlockLoad<float*, BLOCK_THREADS, ITEMS_PER_THREAD,
                         cub::BLOCK_LOAD_WARP_TRANSPOSE>
      BlockLoadT;
  // Specialize BlockStore type for our thread block (uses warp-striped loads
  // for coalescing, then transposes in shared memory to a blocked arrangement).
  typedef cub::BlockStore<float*, BLOCK_THREADS, ITEMS_PER_THREAD,
                          cub::BLOCK_STORE_WARP_TRANSPOSE>
      BlockStoreT;
  // Specialize BlockScan type for our thread block.
  typedef cub::BlockScan<float, BLOCK_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE>
      BlockScanT;

  // Initialize running total.
  BlockPrefixCallbackOp prefix_op(0);

  // Shared memory.
  __shared__ union TempStorage {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;
    typename BlockScanT::TempStorage scan;
  } temp_storage;
  // Per-thread tile data.
  float data[ITEMS_PER_THREAD];

  const int y = blockIdx.y;
  if (y >= 0 && y < image_dim.height) {
    // Have the block iterate over segments of items.
    for (int block_offset = 0; block_offset < image_dim.width;
         block_offset += BLOCK_THREADS * ITEMS_PER_THREAD) {
      // Compute begin/end offsets of our block's row
      const int row_in_offset = y * differential.Pitch() / sizeof(float);

      float* d_in =
          block_offset + row_in_offset + const_cast<float*>(differential.ptr());
      // Load items into a blocked arrangement.

      const int valid_items =
          (image_dim.width - block_offset) < BLOCK_THREADS * ITEMS_PER_THREAD
              ? image_dim.width - block_offset
              : BLOCK_THREADS * ITEMS_PER_THREAD;

      BlockLoadT(temp_storage.load).Load(d_in, data, valid_items);
      // Barrier for smem reuse.
      __syncthreads();
      float block_aggregate;
      // Compute exclusive prefix sum.
      BlockScanT(temp_storage.scan)
          .InclusiveSum(data, data, block_aggregate, prefix_op);

      // Barrier for smem reuse.
      __syncthreads();

      const int row_out_offset = y * integral.Pitch() / sizeof(float);

      float* d_out = block_offset + row_out_offset + integral.ptr();

      // Store items from a blocked arrangement.
      BlockStoreT(temp_storage.store).Store(d_out, data, valid_items);
      // Barrier to co-ordinate along loop tiles.
      __syncthreads();
    }
  }
}

}  // namespace domain_transform

#endif  // CUMSUM_KERNELS2D_H_
