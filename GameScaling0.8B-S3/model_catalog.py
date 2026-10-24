"""Pinned unquantized checkpoints supported by the shared Apple Silicon runner."""
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    revision: str
    weight_bytes: int
    weight_sha256: str
    hidden_size: int

    @property
    def repository(self): return 'THUSI-Lab/' + self.name


MODELS = {
    '0.8b-s3': ModelSpec('0.8b-s3','GameScaling-0.8B-S3',
        '1b89640dcf2ef51fb53ad9338a0725e9622dd200',2214590296,
        'd553c0796833691e32277a937db649eaac39bae9a0028a63c964c12a261c2dfa',1024),
    '2b-s3': ModelSpec('2b-s3','GameScaling-2B-S3',
        '0ef44e6569ebfeb310c5d5a5a973ae30cccba6ac',5443677680,
        '97018239265aed03c16afa0f2acb035a428a3e66195a20518cf3c14dcfe289e8',2048),
}


def resolve_model(value=None):
    value=value or os.environ.get('ANO_GAMESCALING_MODEL','0.8b-s3')
    for spec in MODELS.values():
        if value.lower() in (spec.key,spec.name.lower(),spec.repository.lower()):return spec
    raise ValueError('Choose a supported model: '+', '.join(MODELS))


def data_dir():
    default=Path.home()/'Library/Application Support/AgentNumberOne/gamescaling'
    return Path(os.environ.get('ANO_GAMESCALING_DATA',str(default))).expanduser()


def model_path(spec):
    return Path(os.environ.get('ANO_GAMESCALING_MODEL_PATH',
        str(data_dir()/'models'/(spec.name+'-bf16')))).expanduser()


def validate_config(config,spec):
    if config.get('quantization') or config.get('quantization_config'):
        raise ValueError('This runner requires unquantized source weights')
    if config.get('model_type')!='qwen3_5':raise ValueError('Expected Qwen3.5 checkpoint')
    if config.get('text_config',{}).get('hidden_size')!=spec.hidden_size:
        raise ValueError('Checkpoint architecture differs from selected '+spec.name)
