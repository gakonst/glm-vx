"""Run the actual GGUF CLI, including tokenization, through a small file fixture."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
pytest.importorskip('gguf')
from tokenizers import Tokenizer, models, pre_tokenizers
from test_gguf_checkpoint import fixture_data,write_fixture
from glm_vx.model import Model
from glm_vx.backend import NumpyBackend


def test_generate_cli_reads_gguf_and_emits_matching_tokens(tmp_path):
    config,weights=fixture_data()
    path=write_fixture(tmp_path/'model.gguf',config,weights)
    (tmp_path/'config.json').write_text(json.dumps(config))
    tokenizer=Tokenizer(models.WordLevel({str(i):i for i in range(config['vocab_size'])},unk_token='0'))
    tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path/'tokenizer.json'))
    output=tmp_path/'receipt.json'
    result=subprocess.run([sys.executable,'-m','glm_vx.generate','--gguf',str(path),'--raw','--prompt','1',
                           '--max-tokens','2','--backend','numpy-reference','--output',str(output)],
                          capture_output=True,text=True,timeout=30,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
    assert result.returncode==0,result.stderr
    receipt=json.loads(output.read_text())
    model=Model(config,weights,NumpyBackend());cache=model.new_cache()
    first=int(np.argmax(model.forward(1,cache)));second=int(np.argmax(model.forward(first,cache)))
    assert receipt['status']=='complete'
    assert receipt['generated_token_ids']==[first,second]
    assert receipt['prompt_token_ids']==[1]
    assert receipt['finish_reason']=='length'


def test_chat_prompt_uses_supplied_template(tmp_path):
    pytest.importorskip('jinja2')
    from glm_vx.generate import render_prompt
    (tmp_path/'chat_template.jinja').write_text('{{ messages[0].content }}|{{ reasoning_effort }}|{{ add_generation_prompt }}')
    assert render_prompt(tmp_path,'Hello')=='Hello|low|True'
    assert render_prompt(tmp_path,'raw',raw=True)=='raw'


def test_server_cli_serves_gguf_over_http(tmp_path):
    import http.client
    import select
    import signal
    from urllib.parse import urlsplit
    config,weights=fixture_data()
    path=write_fixture(tmp_path/'model.gguf',config,weights)
    (tmp_path/'config.json').write_text(json.dumps(config))
    process=subprocess.Popen([sys.executable,'-m','glm_vx.server','--gguf',str(path),
                              '--backend','numpy-reference','--port','0'],stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,text=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
    try:
        assert select.select([process.stdout],[],[],10)[0], 'server failed to start'
        ready=process.stdout.readline()
        assert ready,process.stderr.read()
        address=urlsplit(json.loads(ready)['listening'])
        client=http.client.HTTPConnection(address.hostname,address.port,timeout=10)
        client.request('POST','/v1/completions',json.dumps({'prompt':[1],'max_tokens':2}),{'Content-Type':'application/json'})
        response=client.getresponse();body=json.loads(response.read());client.close()
        assert response.status==200,body
        model=Model(config,weights,NumpyBackend());cache=model.new_cache()
        first=int(np.argmax(model.forward(1,cache)));second=int(np.argmax(model.forward(first,cache)))
        assert body['choices'][0]['token_ids']==[first,second]
    finally:
        process.send_signal(signal.SIGINT)
        try:process.communicate(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.communicate(timeout=5)
