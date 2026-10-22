#!/usr/bin/env bash
set -Eeuo pipefail

# The complete upstream LIBERO environment remains an external dependency;
# the release includes derived assets and some shared upstream meshes.
# Optionally set LIBERO_VPRO_ASSET_ARCHIVE (a local .zip/.tar archive) or
# LIBERO_VPRO_ASSET_URL (a user-supplied URL), and choose the extraction
# directory with LIBERO_VPRO_DATA_ROOT.

: "${LIBERO_VPRO_DATA_ROOT:?Set LIBERO_VPRO_DATA_ROOT first}"
python3 - <<'PY'
import os
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile

destination = Path(os.environ['LIBERO_VPRO_DATA_ROOT']).expanduser().resolve()
archive = os.environ.get('LIBERO_VPRO_ASSET_ARCHIVE', '')
url = os.environ.get('LIBERO_VPRO_ASSET_URL', '')
if not archive and not url:
    raise SystemExit('Set LIBERO_VPRO_ASSET_ARCHIVE or LIBERO_VPRO_ASSET_URL.')

def check_member(name):
    target = (destination / name).resolve()
    if Path(name).is_absolute() or not target.is_relative_to(destination):
        raise SystemExit(f'Unsafe archive member: {name}')

with tempfile.TemporaryDirectory(prefix='libero-vpro-assets-') as temporary:
    if url:
        downloaded = Path(temporary) / 'download'
        with urllib.request.urlopen(url, timeout=120) as response, downloaded.open('wb') as handle:
            shutil.copyfileobj(response, handle)
        archive = str(downloaded)
    else:
        archive = str(Path(archive).expanduser())
    # Detect content, not the filename: URLs may have query strings or no suffix.
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                check_member(member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise SystemExit('Symbolic links are not accepted in asset archives.')
            destination.mkdir(parents=True, exist_ok=True)
            bundle.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                check_member(member.name)
                if not (member.isfile() or member.isdir()):
                    raise SystemExit(f'Unsupported archive member: {member.name}')
            destination.mkdir(parents=True, exist_ok=True)
            bundle.extractall(destination)
    else:
        raise SystemExit('Unsupported asset archive: expected ZIP or TAR content.')
print(f'Assets extracted under {destination}')
PY
