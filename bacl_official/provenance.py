import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / 'third_party' / 'BACL'
LOCK_FILE = PROJECT_ROOT / 'third_party' / 'UPSTREAM_LOCK.json'


def verify_upstream():
    lock = json.loads(LOCK_FILE.read_text(encoding='utf-8'))
    failures = []
    for relative, expected in lock['files'].items():
        path = UPSTREAM_ROOT / relative
        if not path.is_file():
            failures.append(relative + ': missing')
            continue
        content = path.read_bytes()
        digest = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if digest != expected:
            failures.append(relative + ': differs from upstream')
    if failures:
        raise RuntimeError('Upstream source verification failed:\n' + '\n'.join(failures))
    return {'repository': lock['repository'], 'commit': lock['commit'],
            'verified_files': len(lock['files'])}


def activate_upstream():
    provenance = verify_upstream()
    path = str(UPSTREAM_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return provenance
