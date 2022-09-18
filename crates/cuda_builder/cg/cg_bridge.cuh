#pragma once
#include "cooperative_groups.h"
namespace cg = cooperative_groups;

typedef struct GridGroupWrapper {
    cg::grid_group gg;
} GridGroupWrapper;

extern "C" typedef void* GridGroup;
extern "C" __device__ GridGroup this_grid();
extern "C" __device__ void GridGroup_destroy(GridGroup gg);
extern "C" __device__ bool GridGroup_is_valid(GridGroup gg);
extern "C" __device__ void GridGroup_sync(GridGroup gg);
extern "C" __device__ unsigned long long GridGroup_size(GridGroup gg);
extern "C" __device__ unsigned long long GridGroup_thread_rank(GridGroup gg);
// extern "C" dim3 GridGroup_group_dim(); // TODO: impl these.
extern "C" __device__ unsigned long long GridGroup_num_threads(GridGroup gg);
// extern "C" dim3 GridGroup_dim_blocks(); // TODO: impl these.
extern "C" __device__ unsigned long long GridGroup_num_blocks(GridGroup gg);
// extern "C" dim3 GridGroup_block_index(); // TODO: impl these.
extern "C" __device__ unsigned long long GridGroup_block_rank(GridGroup gg);
