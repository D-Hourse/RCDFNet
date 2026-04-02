#!/usr/bin/env python3
import importlib
import sys

mods = [
    'torch',
    'mmcv',
    'mmdet',
    'mmseg',
    'rcdfnet',
    'rcdfnet.models',
    'rcdfnet.datasets',
    'rcdfnet.ops',
]

failed = []
print(f'Python: {sys.version}')
for m in mods:
    try:
        importlib.import_module(m)
        print(f'[OK] import {m}')
    except Exception as e:
        failed.append((m, str(e)))
        print(f'[FAIL] import {m}: {e}')

if failed:
    raise SystemExit(1)
print('[OK] environment check passed')
