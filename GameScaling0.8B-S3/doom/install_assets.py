"""Fetch the original DFP Battle-1 scenario with pinned provenance, outside Git."""
import hashlib
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_catalog import data_dir

COMMIT='ea2e9c699b632dcbc423d680abaaf49533c9dfe1'
FILES={
    'maps/D3_battle.cfg': '67b2e6c80da6e0b99ee064150b9e722bb7f63eafcfe1894a6dfd93c5e71b6e25',
    'maps/D3_battle.wad': 'b29b461995ae7035095a5283c57ef13393e6839326754e94f5b80af5f07e5484',
    'LICENSE': '7184574847e62c901747143f8d18bca8d4199764394cc8d6df066cea0e61593b',
}


def install():
    dest=data_dir()/'doom-assets';dest.mkdir(parents=True,exist_ok=True)
    for source,expected in FILES.items():
        target=dest/Path(source).name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==expected:continue
        url=f'https://raw.githubusercontent.com/isl-org/DirectFuturePrediction/{COMMIT}/{source}'
        with urlopen(url,timeout=60) as response:content=response.read()
        if hashlib.sha256(content).hexdigest()!=expected:raise ValueError('Asset SHA-256 differs: '+source)
        temporary=target.with_suffix('.download');temporary.write_bytes(content);temporary.replace(target)
    return dest


if __name__=='__main__':print(install())
