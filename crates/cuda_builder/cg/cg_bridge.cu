#include "cooperative_groups.h"
#include "cg_bridge.cuh"
namespace cg = cooperative_groups;

__device__ GridGroup this_grid()
{
    cg::grid_group gg = cg::this_grid();
    GridGroupWrapper* ggp = new GridGroupWrapper { gg };
    return ggp;
}

__device__ void GridGroup_destroy(GridGroup gg)
{
    GridGroupWrapper* g = static_cast<GridGroupWrapper*>(gg);
    delete g;
}

__device__ bool GridGroup_is_valid(GridGroup gg)
{
    GridGroupWrapper* g = static_cast<GridGroupWrapper*>(gg);
    return g->gg.is_valid();
}

__device__ void GridGroup_sync(GridGroup gg)
{
    GridGroupWrapper* g = static_cast<GridGroupWrapper*>(gg);
    return g->gg.sync();
}

__device__ unsigned long long GridGroup_size(GridGroup gg)
{
    GridGroupWrapper* g = static_cast<GridGroupWrapper*>(gg);
    return g->gg.size();
}

__device__ unsigned long long GridGroup_thread_rank(GridGroup gg)
{
    GridGroupWrapper* g = static_cast<GridGroupWrapper*>(gg);
    return g->gg.thread_rank();
}

__device__ unsigned long long GridGroup_num_threads(GridGroup gg)
{
    GridGroupWrapper* g = static_cast<GridGroupWrapper*>(gg);
    return g->gg.num_threads();
}

__device__ unsigned long long GridGroup_num_blocks(GridGroup gg)
{
    GridGroupWrapper* g = static_cast<GridGroupWrapper*>(gg);
    return g->gg.num_blocks();
}

__device__ unsigned long long GridGroup_block_rank(GridGroup gg)
{
    GridGroupWrapper* g = static_cast<GridGroupWrapper*>(gg);
    return g->gg.block_rank();
}

__host__ int main()
{}
