"""Distribution boundaries: model selection, private files and extraction portability."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'GameScaling0.8B-S3'))
from model_catalog import resolve_model,validate_config
spec=importlib.util.spec_from_file_location('build_release',ROOT/'tools/build_release.py')
bundle=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundle)


class DistributionTests(unittest.TestCase):
    def test_wrong_checkpoint_and_quantization_rejected(self):
        small=resolve_model('THUSI-Lab/GameScaling-0.8B-S3');large=resolve_model('2b-s3')
        config={'model_type':'qwen3_5','text_config':{'hidden_size':small.hidden_size}}
        validate_config(config,small)
        with self.assertRaises(ValueError):validate_config(config,large)
        with self.assertRaises(ValueError):validate_config({**config,'quantization':{'bits':4}},small)

    def test_forbidden_archive_entries(self):
        for name in ('../leak','/tmp/leak','models/a.json','connection.txt','.env',
                     'results/events.jsonl','x/model.safetensors','x/.venv/bin/python','x/key.pem'):
            with self.subTest(name=name),self.assertRaises(ValueError):bundle.validate_name(name)

    @unittest.skipUnless(shutil.which('git'),'Maintainer archive test requires Git')
    def test_bundle_uses_committed_bytes_and_runs_after_extract(self):
        with tempfile.TemporaryDirectory(prefix='kit with spaces ') as directory:
            root=Path(directory)/'repo';root.mkdir()
            def git(*args):return subprocess.check_output(['git',*args],cwd=root,stderr=subprocess.DEVNULL)
            git('init','-q');git('config','user.email','fixture@example.invalid');git('config','user.name','Fixture')
            (root/'hello.py').write_text("print('portable')\n")
            git('add','hello.py');git('commit','-qm','fixture')
            (root/'hello.py').write_text("print('dirty bytes must not ship')\n")
            (root/'connection.txt').write_text('secret-fixture')
            path=Path(directory)/'kit.zip';report=bundle.build(path,root=root)
            with zipfile.ZipFile(path) as archive:
                names=archive.namelist();self.assertEqual(len(names),2)
                manifest=json.loads(archive.read(bundle.PREFIX+'BUNDLE-MANIFEST.json'))
                content=archive.read(bundle.PREFIX+'hello.py')
                self.assertEqual(hashlib.sha256(content).hexdigest(),manifest['files']['hello.py']['sha256'])
                archive.extractall(Path(directory)/'unpacked')
            output=subprocess.check_output([sys.executable,str(Path(directory)/'unpacked'/bundle.PREFIX/'hello.py')],text=True)
            self.assertEqual(output.strip(),'portable');self.assertFalse(manifest['weights_included'])
            second=Path(directory)/'second.zip';self.assertEqual(report['sha256'],bundle.build(second,root=root)['sha256'])


if __name__=='__main__':unittest.main()
