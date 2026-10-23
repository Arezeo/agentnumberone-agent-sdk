"""Download and verify the pinned, unquantized publisher checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path

REPOSITORY = 'THUSI-Lab/GameScaling-0.8B-S3'
REVISION = '1b89640dcf2ef51fb53ad9338a0725e9622dd200'


def main():
    from huggingface_hub import HfApi, snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, default=Path(os.environ.get(
        'ANO_GAMESCALING_DATA', str(Path.home()/'Library/Application Support/AgentNumberOne/gamescaling')))/'models/GameScaling-0.8B-S3-bf16')
    args = parser.parse_args()
    info = HfApi().model_info(REPOSITORY, revision=REVISION, files_metadata=True)
    files = [f.rfilename for f in info.siblings if f.rfilename.endswith(('.json','.jinja','.safetensors'))]
    snapshot_download(REPOSITORY, revision=REVISION, local_dir=args.dest, allow_patterns=files, max_workers=3)
    checksums = {}
    for entry in info.siblings:
        if not entry.rfilename.endswith('.safetensors'):
            continue
        digest = hashlib.sha256()
        with (args.dest/entry.rfilename).open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        expected = entry.lfs.sha256
        if digest.hexdigest() != expected:
            raise ValueError('Weight checksum mismatch: '+entry.rfilename)
        checksums[entry.rfilename] = digest.hexdigest()
    config = json.loads((args.dest/'config.json').read_text())
    assert not config.get('quantization') and not config.get('quantization_config')
    record = {'repository':REPOSITORY,'revision':REVISION,'quantized':False,'trained':False,'sha256':checksums,'source_dtype':'bfloat16'}
    (args.dest/'installation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'verified', 'path':str(args.dest), **record}))


if __name__ == '__main__':
    main()
