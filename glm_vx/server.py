"""Bounded local HTTP token completion service; no third-party serving engine."""
import argparse
import json
import queue
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .scheduler import Scheduler, BusyError

class Server(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, scheduler, tokenizer=None, model_name='glm-vx-tiny-random'):
        super().__init__(address, Handler)
        self.scheduler, self.tokenizer, self.model_name = scheduler, tokenizer, model_name
        self.slots = threading.BoundedSemaphore(32)
    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False): request.close(); return
        try: super().process_request(request,address)
        except BaseException: self.slots.release(); raise
    def process_request_thread(self, request, address):
        try: super().process_request_thread(request,address)
        finally: self.slots.release()

class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def setup(self):
        super().setup()
        self.connection.settimeout(30)
        # SSE tokens are small latency-sensitive writes. Do not let Nagle wait
        # for an ACK before sending a subsequent token.
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    def log_message(self,*args): pass
    def _json(self,code,value):
        data=json.dumps(value,allow_nan=False).encode()
        self.send_response(code)
        if code == 429: self.send_header('Retry-After', '1')
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path=='/health':
            return self._json(200,{'status':'ok','model':self.server.model_name,
                'backend':self.server.scheduler.model.backend.name,**self.server.scheduler.status()})
        if self.path=='/v1/models':
            return self._json(200,{'object':'list','data':[{'id':self.server.model_name,'object':'model','owned_by':'local'}]})
        self._json(404,{'error':'not found'})
    def _sse(self,value):
        self.wfile.write(('data: '+json.dumps(value,allow_nan=False)+'\n\n').encode()); self.wfile.flush()
    def _disconnected(self):
        # Long-running requests must keep the TCP connection open for the reply.
        if not select.select([self.connection], [], [], 0)[0]:return False
        try:return self.connection.recv(1,socket.MSG_PEEK | socket.MSG_DONTWAIT)==b''
        except BlockingIOError:return False
        except OSError:return True
    def do_POST(self):
        req=None; streaming=False
        try:
            if self.path!='/v1/completions':
                self.close_connection=True
                return self._json(404,{'error':'not found'})
            if self.headers.get('Transfer-Encoding'): raise ValueError('chunked request bodies unsupported')
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=1048576: raise ValueError('body must be 1..1048576 bytes')
            raw=self.rfile.read(size)
            if len(raw)!=size: raise ValueError('incomplete body')
            body=json.loads(raw)
            if not isinstance(body,dict): raise ValueError('JSON object required')
            unknown=set(body)-{'model','prompt','max_tokens','temperature','seed','stream'}
            if unknown: raise ValueError('unsupported fields: '+', '.join(sorted(unknown)))
            if body.get('model',self.server.model_name)!=self.server.model_name: raise ValueError('unknown model')
            if type(body.get('stream',False)) is not bool: raise ValueError('stream must be boolean')
            prompt=body.get('prompt')
            if isinstance(prompt,str):
                if self.server.tokenizer is None: raise ValueError('text needs --tokenizer; tiny demo accepts token IDs')
                prompt=self.server.tokenizer.encode(prompt).ids
            req=self.server.scheduler.submit(prompt,body.get('max_tokens',16),body.get('temperature',0),body.get('seed',0))
            ids=[]; created=int(time.time())
            if body.get('stream',False):
                self.send_response(200)
                self.send_header('Content-Type','text/event-stream')
                self.send_header('Cache-Control','no-cache'); self.send_header('Connection','close')
                self.end_headers(); self.close_connection=True; streaming=True
            last_heartbeat=time.monotonic()
            while True:
                if self._disconnected():return
                try: event=req.events.get(timeout=.25)
                except queue.Empty:
                    if streaming and time.monotonic()-last_heartbeat>=10:
                        self.wfile.write(b': keepalive\n\n'); self.wfile.flush()
                        last_heartbeat=time.monotonic()
                    continue
                if event['type']=='token':
                    ids.append(event['token_id'])
                    if streaming:
                        self._sse({'id':req.id,'object':'text_completion','created':created,'model':self.server.model_name,
                            'choices':[{'index':0,'text':'','token_ids':[ids[-1]],'finish_reason':None}]})
                else:
                    if event['reason'] in ('error', 'backpressure'):
                        if streaming: self._sse({'error':event['error']})
                        else: self._json(500,{'error':event['error']})
                    else:
                        text=self.server.tokenizer.decode(ids) if self.server.tokenizer else ''
                        result={'id':req.id,'object':'text_completion','created':created,'model':self.server.model_name,
                            'choices':[{'index':0,'text':text,'token_ids':ids,'finish_reason':event['reason']}],
                            'usage':{'prompt_tokens':event['prompt_tokens'],'completion_tokens':event['completion_tokens'],
                                'total_tokens':event['prompt_tokens']+event['completion_tokens'],
                                'prompt_tokens_details':{'cached_tokens':event['cached_prompt_tokens']}},
                            'timings':event['timings']}
                        if streaming:self._sse(result)
                        else:self._json(200,result)
                    if streaming:self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
                    return
        except BusyError as exc:self._json(429,{'error':str(exc)})
        except (ValueError,TypeError,OverflowError) as exc:
            self.close_connection=True
            if not streaming:self._json(400,{'error':str(exc)})
        except (BrokenPipeError,ConnectionResetError,TimeoutError):pass
        finally:
            if req:self.server.scheduler.cancel(req)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['vx','vx-gpu','numpy-reference'],default='vx')
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--tiny',action='store_true',help='random weights, correctness demo only')
    source.add_argument('--checkpoint',help='local safetensors checkpoint directory')
    source.add_argument('--gguf',help='GLM-DSA GGUF first shard or directory; CPU disk-offload PoC')
    p.add_argument('--config',help='HF config.json matching the GGUF; defaults to config.json beside its shards')
    p.add_argument('--decode-threads',type=int,default=1,help='GGUF unpacking workers, 1..32')
    p.add_argument('--decoded-cache-mib',type=int,default=512,help='bounded GGUF decoded-weight cache')
    p.add_argument('--tokenizer',help='local tokenizer.json')
    p.add_argument('--host',default='127.0.0.1'); p.add_argument('--port',type=int,default=8000)
    p.add_argument('--max-sequences',type=int,default=8); p.add_argument('--token-budget',type=int,default=8192)
    p.add_argument('--prefix-cache-mib',type=int,default=0,help='CPU prompt snapshot storage budget; disabled by default, includes retained Python storage')
    p.add_argument('--max-pending-events',type=int,default=64,help='bounded unread output events per request; slow consumers are terminated')
    p.add_argument('--batched-prefill',action='store_true',help='opt in to bounded layer-wise CPU prefill; one chunk is indivisible')
    p.add_argument('--prefill-chunk',type=int,default=16)
    p.add_argument('--decode-prefill-tokens',type=int,default=4,help='total prefill tokens per round with active decoders; lower favors decode latency')
    p.add_argument('--gpu-ptx-dir',help='directory containing compiled kernels.ptx')
    p.add_argument('--gpu-device',type=int,default=0)
    p.add_argument('--gpu-weight-cache-mib',type=int,default=512)
    p.add_argument('--gpu-tuning',help='verified shape-specific plan produced by python -m gpu.tune')
    args=p.parse_args()
    if args.prefix_cache_mib<0:p.error('prefix cache must be nonnegative')
    if args.max_pending_events<2:p.error('max pending events must be at least two')
    if args.gpu_weight_cache_mib<0:p.error('GPU weight cache must be nonnegative')
    if args.decoded_cache_mib<0:p.error('decoded cache must be nonnegative')
    from .model import Model,tiny_weights
    if args.backend=='vx':
        from kernels.backend import VxBackend
        backend=VxBackend()
    elif args.backend=='vx-gpu':
        from gpu.backend import GPUBackend
        backend=GPUBackend(args.gpu_ptx_dir,args.gpu_device,args.gpu_weight_cache_mib*1024**2,args.gpu_tuning)
    else:
        from .backend import NumpyBackend
        backend=NumpyBackend()
    if args.tiny:
        from .tiny import tiny_config
        config=tiny_config(); weights=tiny_weights(config,seed=7); name='glm-vx-tiny-random'
        if args.backend=='vx-gpu':
            for value in weights.values():value.flags.writeable=False
    elif args.gguf:
        from pathlib import Path
        from .gguf_checkpoint import GGUFCheckpoint
        directory=Path(args.gguf) if Path(args.gguf).is_dir() else Path(args.gguf).parent
        with open(args.config or directory/'config.json') as f:config=json.load(f)
        weights=GGUFCheckpoint(args.gguf,config,cache_bytes=args.decoded_cache_mib*1024**2,decode_threads=args.decode_threads)
        if not args.tokenizer and (directory/'tokenizer.json').exists():args.tokenizer=str(directory/'tokenizer.json')
        name='glm-gguf-vx-experimental'
    else:
        from .checkpoint import Checkpoint
        from .config import GLMConfig
        with open(args.checkpoint+'/config.json') as f:config=json.load(f)
        validated=GLMConfig.from_dict(config)
        weights=Checkpoint(args.checkpoint,block_size=validated.weight_block_size)
        weights.validate_manifest(validated.tensor_shapes(),inspect_shards=True)
        name='glm-5.3-vx-experimental'
    tokenizer=None
    if args.tokenizer:
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(args.tokenizer)
    model=Model(config,weights,backend)
    scheduler=Scheduler(model,max_sequences=args.max_sequences,token_budget=args.token_budget,prefill_chunk=args.prefill_chunk,decode_prefill_tokens=args.decode_prefill_tokens,prefix_cache_bytes=args.prefix_cache_mib*1024**2,max_pending_events=args.max_pending_events,batched_prefill=args.batched_prefill)
    server=Server((args.host,args.port),scheduler,tokenizer,name)
    print(json.dumps({'listening':f'http://{args.host}:{server.server_port}','model':name,'backend':backend.name}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        server.server_close(); scheduler.close()
        if hasattr(weights,'close'):weights.close()
        if hasattr(backend,'close'):backend.close()

if __name__=='__main__':main()
