"""Reproduce ggml_tables.bin from the pinned MIT-licensed ggml-common.h.

Usage: python kernels/extract_ggml_tables.py ../glm-vx-reference
Requires the exact source hash; writes only the tables, provenance and license.
These are codec data tables, not an inference-engine dependency.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

PIN='11fe02151f79c41d0d4af7da708755d73b9c0da6'
SOURCE_HASH='0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c'


def extract(reference,output):
    source=(reference/'ggml/src/ggml-common.h').read_bytes()
    if hashlib.sha256(source).hexdigest()!=SOURCE_HASH:
        raise ValueError('ggml-common.h does not match pinned source')
    text=source.decode();tables={}
    for name,size in [('iq1s_grid',8),('iq2xxs_grid',8),('iq3xxs_grid',4),('ksigns_iq2xs',1),('kvalues_iq4nl',1)]:
        body=text.split(', '+name+',',1)[1].split('GGML_TABLE_END()',1)[0].split('\n',1)[1]
        values=re.findall(r'0x[0-9a-fA-F]+|-?\d+',body)
        tables[name]=b''.join((int(v,0)%(1<<(8*size))).to_bytes(size,'little') for v in values)
    data=b''.join(tables.values())
    meta={'repository':'https://github.com/ggml-org/llama.cpp','commit':PIN,
          'source':'ggml/src/ggml-common.h','source_sha256':SOURCE_HASH,
          'binary_sha256':hashlib.sha256(data).hexdigest(),'tables':{}}
    offset=0
    for name,data_part in tables.items():
        meta['tables'][name]={'offset':offset,'bytes':len(data_part)};offset+=len(data_part)
    output.mkdir(parents=True,exist_ok=True)
    (output/'ggml_tables.bin').write_bytes(data)
    (output/'ggml_tables.json').write_text(json.dumps(meta,indent=2)+'\n')
    (output/'GGML-LICENSE').write_bytes((reference/'LICENSE').read_bytes())

if __name__=='__main__':extract(Path(sys.argv[1]),Path(__file__).parent)
