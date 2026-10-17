"""Windows/Linux dataset recovery; checks upstream fixed MD5 before safe extraction."""
import argparse
import hashlib
from pathlib import Path
import urllib.request
import zipfile
FILES={'CRC-VAL-HE-7K.zip':'2fd1651b4f94ebd818ebf90ad2b6ce06','NCT-CRC-HE-100K.zip':'6fd702d11df6292bc054397ae038a464'}

def md5(path):
    h=hashlib.md5()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('data/raw'));a=p.parse_args()
    zdir=a.root/'zips';zdir.mkdir(parents=True,exist_ok=True)
    for name,checksum in FILES.items():
        dest=zdir/name
        if not dest.exists() or md5(dest)!=checksum:
            part=dest.with_suffix('.part')
            # Range resume only when the server explicitly honors the Range request.
            size=part.stat().st_size if part.exists() else 0
            req=urllib.request.Request(f'https://zenodo.org/records/1214456/files/{name}?download=1',headers={'Range':f'bytes={size}-'} if size else {})
            with urllib.request.urlopen(req,timeout=120) as response:
                mode='ab' if size and response.status==206 else 'wb'
                with part.open(mode) as f:
                    while data:=response.read(2**20):f.write(data)
            if md5(part)!=checksum:raise ValueError(f'{name}: checksum mismatch; retain .part for inspection/retry')
            part.replace(dest)
        with zipfile.ZipFile(dest) as z:
            for item in z.infolist():
                target=(a.root/item.filename).resolve()
                if not target.is_relative_to(a.root.resolve()):raise ValueError('Unsafe archive path')
                if target.exists() and target.is_file() and target.stat().st_size==item.file_size:continue
                z.extract(item,a.root)
        print(f'Verified and restored {name}')
if __name__=='__main__':main()
