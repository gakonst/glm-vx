// Test-only CPU SIMT emulator. GPU production runtime never links this file.
// Runs exactly the Vx kernel object under 32 host threads per warp. This tests
// lane/reduction arithmetic; it cannot establish GPU codegen or race behavior.
#include <barrier>
#include <thread>
#include <vector>
#include <memory>
#include <cstdint>
#include <cmath>
#include <bit>
struct Warp {
    std::barrier<> sync{32};
    float floats[32]{};
    float mma_a[16][8]{};
    float mma_b[8][8]{};
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
// Test-only equation model of PTX lane fragments; not hardware MMA rounding.
static float tf32_rna(float value) {
    uint32_t bits=std::bit_cast<uint32_t>(value);
    if ((bits & 0x7f800000u)==0x7f800000u) return value;
    return std::bit_cast<float>((bits+0x1000u)&0xffffe000u);
}
extern "C" void vx_gpu_mma_tf32_m16n8k8(float* out, int32_t offset,
    float a0,float a1,float a2,float a3,float b0,float b1,
    float c0,float c1,float c2,float c3) {
    int lane=emu_tid%32, g=lane/4, t=lane%4;
    auto& w=*emu_warp;
    w.mma_a[g][t]=tf32_rna(a0); w.mma_a[g+8][t]=tf32_rna(a1);
    w.mma_a[g][t+4]=tf32_rna(a2); w.mma_a[g+8][t+4]=tf32_rna(a3);
    w.mma_b[t][g]=tf32_rna(b0); w.mma_b[t+4][g]=tf32_rna(b1);
    w.sync.arrive_and_wait();
    float acc[4]={c0,c1,c2,c3};
    for (int i=0;i<4;i++) {
        int row=g+(i/2)*8, col=t*2+i%2;
        for (int k=0;k<8;k++) acc[i]=std::fma(w.mma_a[row][k],w.mma_b[k][col],acc[i]);
        out[offset+i]=acc[i];
    }
    w.sync.arrive_and_wait();
}
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
