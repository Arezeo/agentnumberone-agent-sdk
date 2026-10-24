"""Download and verify the pinned, unquantized publisher checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from model_catalog import MODELS, resolve_model, data_dir, validate_config

def main():
    from huggingface_hub import HfApi, snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, default=os.environ.get('ANO_GAMESCALING_MODEL','0.8b-s3'))
    parser.add_argument('--dest', type=Path)
    args = parser.parse_args()
    spec=resolve_model(args.model)
    args.dest=args.dest or data_dir()/'models'/(spec.name+'-bf16')
    info = HfApi().model_info(spec.repository, revision=spec.revision, files_metadata=True)
    files = [f.rfilename for f in info.siblings if f.rfilename.endswith(('.json','.jinja','.safetensors'))]
    snapshot_download(spec.repository, revision=spec.revision, local_dir=args.dest, allow_patterns=files, max_workers=3)
    if (args.dest/'model.safetensors').stat().st_size!=spec.weight_bytes:
        raise ValueError('Weight byte count differs from pinned checkpoint')
    checksums = {}
    for entry in info.siblings:
        if not entry.rfilename.endswith('.safetensors'):
            continue
        digest = hashlib.sha256()
        with (args.dest/entry.rfilename).open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        expected = entry.lfs.sha256
        if entry.rfilename=='model.safetensors' and expected!=spec.weight_sha256:
            raise ValueError('Publisher metadata differs from pinned weight digest')
        if digest.hexdigest() != expected:
            raise ValueError('Weight checksum mismatch: '+entry.rfilename)
        checksums[entry.rfilename] = digest.hexdigest()
    config = json.loads((args.dest/'config.json').read_text())
    validate_config(config,spec)
    record = {'repository':spec.repository,'revision':spec.revision,'quantized':False,'trained':False,'sha256':checksums,'source_dtype':'bfloat16'}
    (args.dest/'installation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'verified', 'path':str(args.dest), **record}))


if __name__ == '__main__':
    main()
