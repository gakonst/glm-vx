// Test-only CPU SIMT emulator. GPU production runtime never links this file.
// Runs exactly the Vx kernel object under 32 host threads per warp. This tests
// lane/reduction arithmetic; it cannot establish GPU codegen or race behavior.
#include <barrier>
#include <thread>
#include <vector>
#include <memory>
#include <cstdint>
#include <cmath>
struct Warp {
    std::barrier<> sync{32};
    float floats[32]{};
    int32_t integers[32]{};
};
thread_local int32_t emu_tid, emu_bid, emu_block, emu_grid;
thread_local std::barrier<>* emu_barrier;
thread_local float* emu_shared;
thread_local Warp* emu_warp;
extern "C" int32_t vx_thread_idx(){return emu_tid;}
extern "C" int32_t vx_block_idx(){return emu_bid;}
extern "C" int32_t vx_block_dim(){return emu_block;}
extern "C" float vx_shfl_down(float value,int32_t delta){
    int lane=emu_tid%32;
    emu_warp->floats[lane]=value;
    emu_warp->sync.arrive_and_wait();
    float result=emu_warp->floats[lane+delta<32?lane+delta:lane];
    emu_warp->sync.arrive_and_wait();
    return result;
}
extern "C" int32_t vx_shfl_down_i32(int32_t value,int32_t delta){
    int lane=emu_tid%32;
    emu_warp->integers[lane]=value;
    emu_warp->sync.arrive_and_wait();
    int32_t result=emu_warp->integers[lane+delta<32?lane+delta:lane];
    emu_warp->sync.arrive_and_wait();
    return result;
}
extern "C" float vx_shfl(float value,int32_t source){
    int lane=emu_tid%32;
    emu_warp->floats[lane]=value;
    emu_warp->sync.arrive_and_wait();
    float result=emu_warp->floats[source];
    emu_warp->sync.arrive_and_wait();
    return result;
}
extern "C" int32_t vx_shfl_i32(int32_t value,int32_t source){
    int lane=emu_tid%32;
    emu_warp->integers[lane]=value;
    emu_warp->sync.arrive_and_wait();
    int32_t result=emu_warp->integers[source];
    emu_warp->sync.arrive_and_wait();
    return result;
}
extern "C" float vx_exp(float x){return std::exp(x);}
extern "C" float vx_sqrt(float x){return std::sqrt(x);}
extern "C" float vx_rsqrt(float x){return 1.0f/std::sqrt(x);}
extern "C" int32_t vx_gpu_thread_idx_x(){return emu_tid;}
extern "C" int32_t vx_gpu_block_idx_x(){return emu_bid;}
extern "C" int32_t vx_gpu_block_dim_x(){return emu_block;}
extern "C" int32_t vx_gpu_grid_dim_x(){return emu_grid;}
extern "C" float vx_gpu_shuffle_down_f32(float v,int32_t d){return vx_shfl_down(v,d);}
extern "C" float vx_gpu_shuffle_idx_f32(float v,int32_t d){return vx_shfl(v,d);}
extern "C" int32_t vx_gpu_shuffle_down_i32(int32_t v,int32_t d){return vx_shfl_down_i32(v,d);}
extern "C" int32_t vx_gpu_shuffle_idx_i32(int32_t v,int32_t d){return vx_shfl_i32(v,d);}
extern "C" void vx_gpu_barrier(){emu_barrier->arrive_and_wait();}
extern "C" float* vx_gpu_shared_f32(){return emu_shared;}
extern "C" int32_t simt_launch(void (*entry)(void*),void* arg,int32_t grid,int32_t block){
    if(grid<1||block<32||block%32||block>256)return -1;
    for(int b=0;b<grid;b++){
        std::barrier<> barrier(block);
        float shared[4096]{};
        std::vector<std::unique_ptr<Warp>> warps;
        for(int w=0;w<block/32;w++)warps.emplace_back(new Warp());
        std::vector<std::thread> lanes;
        for(int t=0;t<block;t++)lanes.emplace_back([&,t]{
            emu_tid=t;emu_bid=b;emu_block=block;emu_grid=grid;
            emu_warp=warps[t/32].get();emu_barrier=&barrier;emu_shared=shared;
            entry(arg);
        });
        for(auto& lane:lanes)lane.join();
    }
    return 0;
}
