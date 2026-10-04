// Independent test-only oracle. Build against pinned llama.cpp, never linked by serving.
#include "llama.h"
#include "ggml-backend.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
struct Capture { std::string dir; int position = 0; bool failed = false; };
static bool capture(ggml_tensor * t, bool ask, void * data) {
    auto & c = *static_cast<Capture *>(data);
    std::string name(t->name);
    bool wanted = name.rfind("l_out-", 0)==0;
    if (ask) return wanted;
    if (!wanted) return true;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) { c.failed=true; return false; }
    std::vector<float> values(ggml_nelements(t));
    ggml_backend_tensor_get(t, values.data(), 0, values.size()*sizeof(float));
    std::ofstream f(c.dir + "/p" + std::to_string(c.position) + "-" + name + ".f32", std::ios::binary);
    f.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float));
    if (!f) {c.failed=true; return false;} return true;
}
int main(int argc, char **argv) {
    if (argc < 4) {std::cerr<<"usage: llama-trace model.gguf output-directory token-id...\n";return 2;}
    try {
        Capture c{argv[2]}; if(std::filesystem::exists(c.dir)) throw std::runtime_error("output already exists");
        std::vector<llama_token> tokens; for(int i=3;i<argc;i++) tokens.push_back(std::stoi(argv[i]));
        llama_backend_init(); auto mp=llama_model_default_params();mp.n_gpu_layers=0;
        auto *model=llama_model_load_from_file(argv[1],mp);if(!model)throw std::runtime_error("model load failed");
        int nv=llama_vocab_n_tokens(llama_model_get_vocab(model));
        for(auto t:tokens)if(t<0||t>=nv)throw std::runtime_error("token outside vocabulary");
        std::filesystem::create_directories(c.dir);
        auto cp=llama_context_default_params();cp.n_ctx=256;cp.n_batch=1;cp.n_ubatch=1;cp.n_threads=4;cp.n_threads_batch=4;cp.cb_eval=capture;cp.cb_eval_user_data=&c;
        if(tokens.size()>256)throw std::runtime_error("this bounded exporter supports at most256 tokens");
        auto *ctx=llama_init_from_model(model,cp);if(!ctx)throw std::runtime_error("context init failed");
        auto *batch=llama_batch_ext_init(ctx);
        for(size_t p=0;p<tokens.size();p++) {
            c.position=p;llama_batch_ext_clear(batch);int i=llama_batch_ext_add_token(batch,0,tokens[p]);llama_pos pos=p;
            llama_batch_ext_set_pos(batch,i,&pos);llama_batch_ext_set_output_logits(batch,i,true);
            if(llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,batch)||c.failed)throw std::runtime_error("decode/trace failed");
            auto *logits=llama_get_logits_ith(ctx,-1);if(!logits)throw std::runtime_error("missing logits");
            std::ofstream f(c.dir+"/p"+std::to_string(p)+"-logits.f32",std::ios::binary);
            f.write(reinterpret_cast<const char *>(logits),nv*sizeof(float));if(!f)throw std::runtime_error("write failed");
            std::cerr<<"trace position "<<p<<" complete\n";
        }
        std::ofstream done(c.dir+"/complete.txt");done<<tokens.size()<<" "<<nv<<"\n";done.close();if(!done)throw std::runtime_error("receipt failed");
        llama_batch_ext_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}
}
