import importlib.util
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import pytest

spec = importlib.util.spec_from_file_location('serving_bench', Path(__file__).parents[1]/'benchmarks/serve.py')
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)

@pytest.mark.parametrize('ending', ['valid','truncated','error','mismatch'])
def test_http_stream_counts_only_incremental_tokens_and_requires_completion(ending):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
            def send(x): self.wfile.write(('data: '+json.dumps(x)+'\n\n').encode()); self.wfile.flush()
            for token in [2,3]: send({'choices':[{'token_ids':[token], 'finish_reason':None}]})
            if ending=='truncated': return
            if ending=='error': send({'error':'failure'}); return
            send({'choices':[{'token_ids':[2,3] if ending=='valid' else [5], 'finish_reason':'length'}]})
            self.wfile.write(b'data: [DONE]\n\n')
    server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=server.serve_forever); thread.start()
    try:
        result = bench.completion(f'http://127.0.0.1:{server.server_port}', [1],2,timeout=3)
        assert result['ok'] == (ending=='valid')
        assert result['output_tokens']==2
        if result['ok']:
            assert len(result['inter_token_ms'])==1
            assert result['latency_ms']>=result['ttft_ms']>=0
    finally: server.shutdown(); server.server_close(); thread.join()


def test_aggregate_excludes_failed_request_tokens_but_reports_failure():
    rows = [{'ok':True,'output_tokens':2,'ttft_ms':4,'inter_token_ms':[2],'latency_ms':7},
            {'ok':False,'output_tokens':1,'latency_ms':9}]
    result=bench.summarize(rows,2)
    assert result['output_tokens_per_second']==1
    assert result['failed_requests']==1
    assert result['ttft_ms']['p95']==4
    assert bench.percentiles([])['p50'] is None
