"""Cast original SafeTensors to FP16 in bounded CPU chunks, without quantization.

Keeps source keys and shapes for the normal MLX loader's sanitization. Does not
load either complete checkpoint into RAM. Original files remain untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import time

import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('--source',type=Path,required=True)
ap.add_argument('--dest',type=Path,required=True)
args=ap.parse_args()
source=args.source.resolve();dest=args.dest.resolve()
assert source!=dest
weight=source/'model.safetensors'
dest.mkdir(parents=True,exist_ok=True)
target=dest/'model.safetensors'
assert not target.exists(), 'Destination weights already exist'
with weight.open('rb') as stream:
    header_size=struct.unpack('<Q',stream.read(8))[0]
    original=json.loads(stream.read(header_size))
data_start=8+header_size
entries=sorted(((name,entry) for name,entry in original.items() if name!='__metadata__'),key=lambda item:item[1]['data_offsets'][0])
header={'__metadata__':original.get('__metadata__',{'format':'pt'})}
offset=0
sizes={'BF16':2,'F16':2,'F32':4,'F64':8,'I64':8,'I32':4,'I16':2,'I8':1,'U8':1,'BOOL':1}
for name,entry in entries:
    dtype=entry['dtype'];start,end=entry['data_offsets']
    assert dtype in sizes, f'Unsupported dtype {dtype}'
    count=(end-start)//sizes[dtype]
    output_dtype='F16' if dtype in ('BF16','F16','F32','F64') else dtype
    size=count*sizes[output_dtype]
    header[name]={'dtype':output_dtype,'shape':entry['shape'],'data_offsets':[offset,offset+size]}
    offset+=size
encoded=json.dumps(header,separators=(',',':')).encode()
encoded+=b' '*((-len(encoded))%8)
prefix=struct.pack('<Q',len(encoded))+encoded
digest=hashlib.sha256();temporary=dest/'model.safetensors.incomplete'
started=time.time();converted=0
with weight.open('rb') as reader,temporary.open('wb') as writer:
    writer.write(prefix);digest.update(prefix)
    for index,(name,entry) in enumerate(entries):
        start,end=entry['data_offsets'];reader.seek(data_start+start);remaining=end-start
        while remaining:
            chunk=reader.read(min(16*1024*1024,remaining));assert chunk
            remaining-=len(chunk)
            if entry['dtype']=='BF16':
                values=(np.frombuffer(chunk,dtype='<u2').astype('<u4')<<16).view('<f4')
            elif entry['dtype'] in ('F16','F32','F64'):
                values=np.frombuffer(chunk,dtype={'F16':'<f2','F32':'<f4','F64':'<f8'}[entry['dtype']])
            else:values=None
            if values is not None:
                with np.errstate(over='ignore',invalid='ignore'):
                    result=values.astype('<f2')
                assert np.isfinite(result).all(), f'Non-finite FP16 conversion: {name}'
                chunk=result.tobytes()
            writer.write(chunk);digest.update(chunk);converted+=len(chunk)
        if index%40==0:
            print(json.dumps({'tensor':index+1,'of':len(entries),'bytes_written':converted,'seconds':time.time()-started}),flush=True)
assert temporary.stat().st_size==len(prefix)+offset
temporary.replace(target)
for path in source.iterdir():
    if path.suffix in ('.json','.jinja'):
        shutil.copy2(path,dest/path.name)
config=json.loads((dest/'config.json').read_text())
assert not config.get('quantization') and not config.get('quantization_config')
def set_dtype(value):
    if isinstance(value,dict):
        for key,child in value.items():
            if key in ('dtype','torch_dtype'):value[key]='float16'
            else:set_dtype(child)
    elif isinstance(value,list):
        for child in value:set_dtype(child)
set_dtype(config)
(dest/'config.json').write_text(json.dumps(config,indent=2))
report={'source':str(source),'destination':str(dest),'operation':'BF16 to IEEE FP16 cast in bounded CPU chunks','quantized':False,'trained':False,'weight_bytes':target.stat().st_size,'sha256':digest.hexdigest(),'tensor_count':len(entries),'source_dtypes':sorted({entry['dtype'] for _,entry in entries}),'output_dtypes':sorted({entry['dtype'] for name,entry in header.items() if name!='__metadata__'}),'all_float_values_finite':True,'seconds':time.time()-started}
(dest/'fp16-provenance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
