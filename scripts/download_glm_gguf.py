#!/usr/bin/env python3
"""Download the pinned full GLM-5.3 IQ1_S checkpoint with resume and SHA256 checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import urllib.request

REPO='unsloth/GLM-5.3-GGUF'
REVISION='346b3591c7f28d1a23716f97a065ecf12ec14771'
PREFIX='UD-IQ1_S/'
TOKENIZER_REVISION='aca966e4e02791568aa6a4ced368624b3d897f42'
ASSETS={
    'config.json':'3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69',
    'tokenizer.json':'19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d',
    'tokenizer_config.json':'98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc',
    'chat_template.jinja':'3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c',
}

def fetch_assets(root):
    provenance={}
    for name,expected in ASSETS.items():
        dest=root/name
        if dest.exists():data=dest.read_bytes()
        else:
            url=f'https://huggingface.co/zai-org/GLM-5.3/resolve/{TOKENIZER_REVISION}/{name}'
            with urllib.request.urlopen(url,timeout=120) as response:data=response.read(32*1024**2+1)
        if len(data)>32*1024**2 or hashlib.sha256(data).hexdigest()!=expected:
            raise RuntimeError(name+': asset size/hash mismatch; existing files are not overwritten')
        if not dest.exists():
            tmp=root/(name+'.part');tmp.write_bytes(data);tmp.replace(dest)
        provenance[name]={'bytes':len(data),'sha256':expected}
    (root/'tokenizer-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    if not 1<=a.workers<=6:p.error('workers must be 1..6')
    root=a.directory;root.mkdir(parents=True,exist_ok=True)
    fetch_assets(root)
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true',timeout=60) as r:meta=json.load(r)
    files=[f for f in meta['siblings'] if f['rfilename'].startswith(PREFIX) and f['rfilename'].endswith('.gguf')]
    if len(files)!=6:raise RuntimeError('pinned manifest must contain exactly six shards')
    manifest={'repo':REPO,'revision':REVISION,'files':files}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    remaining=sum(max(0,f['size']-max((root/Path(f['rfilename']).name).stat().st_size if (root/Path(f['rfilename']).name).exists() else 0,(root/(Path(f['rfilename']).name+'.part')).stat().st_size if (root/(Path(f['rfilename']).name+'.part')).exists() else 0)) for f in files)
    if shutil.disk_usage(root).free < remaining+10*1024**3:raise RuntimeError('insufficient disk space including 10GiB safety margin')
    lock=threading.Lock();status={};last=[0.]
    def update(name,done,state,force=False):
        with lock:
            status[name]={'bytes':done,'state':state}
            if force or time.monotonic()-last[0]>2:
                tmp=root/'download-status.json.tmp';tmp.write_text(json.dumps({'updated':time.time(),'files':status},indent=2)+'\n');tmp.replace(root/'download-status.json');last[0]=time.monotonic()
    def download(f):
        name=Path(f['rfilename']).name;dest=root/name;part=root/(name+'.part');size=f['size'];expected=f['lfs']['sha256']
        if dest.exists():
            if dest.stat().st_size!=size:raise RuntimeError(name+': existing complete file has wrong size')
            source=dest
        else:source=part
        for attempt in range(3):
            try:
                done=source.stat().st_size if source.exists() else 0
                if done>size:raise RuntimeError(name+': oversized partial')
                h=hashlib.sha256()
                if done:
                    update(name,done,'verifying prefix',True)
                    with source.open('rb') as r:
                        for chunk in iter(lambda:r.read(8*1024**2),b''):h.update(chunk)
                if done<size:
                    headers={'Range':f'bytes={done}-'} if done else {}
                    req=urllib.request.Request(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{f["rfilename"]}',headers=headers)
                    with urllib.request.urlopen(req,timeout=120) as r,part.open('ab' if done else 'wb') as out:
                        if done and (r.status!=206 or r.headers.get('Content-Range')!=f'bytes {done}-{size-1}/{size}'):raise RuntimeError(name+': server did not honor resume range')
                        while True:
                            chunk=r.read(8*1024**2)
                            if not chunk:break
                            if done+len(chunk)>size:raise RuntimeError(name+': response exceeds expected size')
                            if shutil.disk_usage(root).free<10*1024**3:raise RuntimeError('disk safety margin reached')
                            out.write(chunk);h.update(chunk);done+=len(chunk);update(name,done,'downloading')
                if done!=size or h.hexdigest()!=expected:raise RuntimeError(name+': size/SHA256 mismatch')
                if source!=dest:part.replace(dest)
                update(name,done,'verified',True)
                print(name+' verified',flush=True);return
            except Exception as exc:
                update(name,source.stat().st_size if source.exists() else 0,type(exc).__name__,True)
                if attempt==2:raise
                time.sleep(2**attempt)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(download,files))
    print('All six shards verified against pinned SHA256 manifest.',flush=True)

if __name__=='__main__':main()
