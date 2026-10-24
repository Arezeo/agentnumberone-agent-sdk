#!/usr/bin/env python3
"""Build a source-only, checksum-manifest ZIP from a committed Git revision."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PREFIX='agentnumberone-gamescaling-kit/'
FORBIDDEN_SUFFIXES={'.safetensors','.gguf','.pth','.pt','.onnx','.pem','.key','.log','.lmp','.mp4'}


def validate_name(name):
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe archive path: '+name)
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:raise ValueError('Non-source file: '+name)
    if any(p in ('models','results','recordings','connection.txt','.env') or p.startswith(('.venv','.env.')) for p in path.parts):
        raise ValueError('Private/runtime file: '+name)


def build(output,ref='HEAD',root=ROOT):
    commit=subprocess.check_output(['git','rev-parse',ref+'^{commit}'],cwd=root,text=True).strip()
    source=subprocess.check_output(['git','archive','--format=zip',commit],cwd=root)
    files={};payloads={}
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        for item in archive.infolist():
            if item.is_dir():continue
            validate_name(item.filename)
            # Reject symlinks: an extracted source bundle must not reference local files.
            if ((item.external_attr>>16)&0o170000)==0o120000:raise ValueError('Symlink: '+item.filename)
            content=archive.read(item)
            if len(content)>2*1024*1024:raise ValueError('Unexpected large source file: '+item.filename)
            files[item.filename]={'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
            payloads[item.filename]=content
    manifest={'repository':'https://github.com/Arezeo/agentnumberone-agent-sdk',
        'source_commit':commit,'weights_included':False,'files':files}
    payloads['BUNDLE-MANIFEST.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,content in sorted(payloads.items()):
            entry=zipfile.ZipInfo(PREFIX+name,date_time=(2026,1,1,0,0,0))
            entry.compress_type=zipfile.ZIP_DEFLATED;entry.external_attr=0o100644<<16
            archive.writestr(entry,content)
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix+'.sha256').write_text(digest+'  '+output.name+'\n')
    return {'zip':str(output),'sha256':digest,'source_commit':commit,'files':len(files),'bytes':output.stat().st_size}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref',default='HEAD');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.output,args.ref),indent=2))
