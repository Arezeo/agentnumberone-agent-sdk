#!/usr/bin/env python3
"""Portable entry point for the unquantized 0.8B/2B Apple Silicon examples."""
import argparse
import fcntl
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parent
EXAMPLE=ROOT/'GameScaling0.8B-S3'
sys.path.insert(0,str(EXAMPLE))
from model_catalog import MODELS, data_dir


def python():
    override=os.environ.get('ANO_GAMESCALING_PYTHON')
    path=Path(override) if override else ROOT/'.venv-gamescaling/bin/python'
    if not path.is_file():raise SystemExit('Run: python3 gamescaling.py setup --model 0.8b-s3')
    return str(path)


def require_mac():
    if platform.system()!='Darwin' or platform.machine()!='arm64':
        raise SystemExit('MLX inference requires an Apple Silicon Mac. CPU transport tests are portable.')


def require_idle(port):
    with socket.socket() as sock:
        sock.settimeout(.3)
        if sock.connect_ex(('127.0.0.1',port))==0:
            raise SystemExit(f'Port {port} has a listener. Stop the current model before another GPU test.')


def public_url(value):
    parsed=urlsplit(value)
    if parsed.scheme!='wss' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path!='/control':
        raise argparse.ArgumentTypeError('Use a trusted wss://HOST[:PORT]/control URL, without credentials or token')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    setup=commands.add_parser('setup',help='Create environment, install pinned dependencies and fetch verified weights')
    setup.add_argument('--model',choices=MODELS,default='0.8b-s3')
    setup.add_argument('--no-download',action='store_true')
    setup.add_argument('--with-doom',action='store_true',help='Also install the optional ViZDoom engine')
    run=commands.add_parser('run',help='Run native model, WebRTC controls and local video viewer')
    run.add_argument('--model',choices=MODELS,default='0.8b-s3')
    run.add_argument('--profile',choices=('general','pixel-duel','crossroad'),default='general')
    run.add_argument('--public-url',required=True,type=public_url,help='Trusted wss:// endpoint without a token')
    run.add_argument('--origin',default='https://agentnumberone.io')
    run.add_argument('--port',type=int,default=8767)
    for command in ('doctor','smoke','webrtc-smoke','profile','doom'):
        sub=commands.add_parser(command)
        sub.add_argument('--model',choices=MODELS,default='0.8b-s3')
        if command=='doctor':sub.add_argument('--verify-weights',action='store_true')
        elif command in ('smoke','webrtc-smoke'):
            if command=='smoke':sub.add_argument('--image',type=Path)
            sub.add_argument('--output',type=Path,required=True)
        elif command=='profile':
            sub.add_argument('--frames',type=Path,nargs='+',required=True)
            sub.add_argument('--output',type=Path,required=True)
        else:
            sub.add_argument('--output',type=Path,required=True)
            sub.add_argument('--max-decisions',type=int,default=10)
            sub.add_argument('--mode',choices=('realtime','paused'),default='realtime')
    commands.add_parser('test',help='CPU contracts and real WebRTC loopback, without model weights')
    args=parser.parse_args()
    env={**os.environ,'PYTHONUNBUFFERED':'1'}
    env['ANO_GAMESCALING_PRECISION']='fp16'
    if hasattr(args,'model'):env['ANO_GAMESCALING_MODEL']=args.model
    if args.command=='setup':
        require_mac()
        if sys.version_info[:2]!=(3,12):raise SystemExit('Use Python 3.12 for the pinned environment.')
        if not (ROOT/'.venv-gamescaling/bin/python').exists():
            subprocess.run([sys.executable,'-m','venv',str(ROOT/'.venv-gamescaling')],check=True)
        subprocess.run([python(),'-m','pip','install','-r',str(EXAMPLE/'requirements.txt')],check=True)
        if args.with_doom:
            subprocess.run([python(),'-m','pip','install','-r',str(EXAMPLE/'doom/requirements.txt')],check=True)
        if not args.no_download:
            subprocess.run([python(),str(EXAMPLE/'install.py'),'--model',args.model],env=env,check=True)
        return
    if args.command=='test':
        subprocess.run([python(),'-m','unittest','discover','-s',str(EXAMPLE/'tests'),'-v'],cwd=ROOT,env=env,check=True)
        subprocess.run([python(),str(ROOT/'tests/webrtc-client.test.py')],cwd=ROOT,env=env,check=True)
        subprocess.run([python(),'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_*.py','-v'],cwd=ROOT,env=env,check=True)
        return
    if args.command=='doctor':
        command=[python(),str(EXAMPLE/'doctor.py')]
        if args.verify_weights:command+=['--verify-weights']
    else:
        require_mac();require_idle(getattr(args,'port',8767))
        data_dir().mkdir(parents=True,exist_ok=True)
        lock=(data_dir()/'inference.lock').open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another kit model owns the GPU. Stop it before this test.')
        if args.command=='run':
            env.update(ANO_GAMESCALING_PRECISION='fp16',ANO_GAMESCALING_PROMPT_PROFILE=args.profile,
                ANO_GAMESCALING_PUBLIC_URL=args.public_url,ANO_GAMESCALING_ORIGIN=args.origin,
                ANO_GAMESCALING_PORT=str(args.port))
            command=[python(),str(EXAMPLE/'webrtc_live.py')]
        elif args.command=='smoke':
            command=[python(),str(EXAMPLE/'smoke.py'),'--output',str(args.output)]
            if args.image:command+=['--image',str(args.image)]
        elif args.command=='webrtc-smoke':
            command=[python(),str(EXAMPLE/'webrtc_smoke.py'),'--output',str(args.output)]
        elif args.command=='profile':
            command=[python(),str(EXAMPLE/'profile_runtime.py'),'--frames',*map(str,args.frames),'--output',str(args.output)]
        else:
            command=[python(),str(EXAMPLE/'doom/run.py'),'--output',str(args.output),
                     '--max-decisions',str(args.max_decisions),'--mode',args.mode]
    result=subprocess.run(command,cwd=ROOT,env=env)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
