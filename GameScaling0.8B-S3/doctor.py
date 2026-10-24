"""Inspect environment/provenance without loading a model or printing pairing secrets."""
import argparse
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import sys

from model_catalog import resolve_model,model_path,validate_config,data_dir


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-weights',action='store_true')
    args=parser.parse_args();spec=resolve_model();path=model_path(spec)
    errors=[];packages={}
    for name in ('mlx','mlx-vlm','transformers','huggingface-hub','aiortc','aiohttp','numpy','Pillow'):
        try:packages[name]=version(name)
        except PackageNotFoundError:errors.append('Missing package: '+name)
    if platform.system()!='Darwin' or platform.machine()!='arm64':errors.append('Inference requires Apple Silicon macOS')
    if sys.version_info[:2]!=(3,12):errors.append('Pinned environment requires Python 3.12')
    verified=False
    try:
        validate_config(json.loads((path/'config.json').read_text()),spec)
        weight=path/'model.safetensors'
        if weight.stat().st_size!=spec.weight_bytes:errors.append('Original checkpoint byte count differs')
        if args.verify_weights:
            digest=hashlib.sha256()
            with weight.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
            verified=digest.hexdigest()==spec.weight_sha256
            if not verified:errors.append('Original checkpoint SHA-256 differs')
    except (OSError,ValueError) as error:errors.append(str(error))
    upstream=Path(__file__).resolve().parent/'upstream'
    provenance=json.loads((upstream/'SOURCE.json').read_text())
    for name,expected in provenance['files'].items():
        if hashlib.sha256((upstream/name).read_bytes()).hexdigest()!=expected:
            errors.append('Vendored upstream file changed: '+name)
    result={'model':spec.name,'repository':spec.repository,'revision':spec.revision,
        'source_weight_bytes':spec.weight_bytes,'full_weight_sha256_verified':verified,
        'runtime_precision':'fp16 cast from original bf16; no quantization',
        'data_dir':str(data_dir()),'model_path':str(path),'packages':packages,
        'upstream_commit':provenance['commit'],'errors':errors,'ready':not errors}
    print(json.dumps(result,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
