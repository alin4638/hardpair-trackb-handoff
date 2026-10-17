"""Export all nonignored Track B source, including untracked files, plus a complete patch."""
from pathlib import Path
import subprocess
import zipfile
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent

def main():
    out=ROOT/'outputs/handoff';out.mkdir(parents=True,exist_ok=True)
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','--','trackb'],cwd=REPO,text=True).splitlines()
    files=sorted({n for n in names if (REPO/n).is_file()})
    patch=subprocess.check_output(['git','diff','--binary','HEAD','--','trackb'],cwd=REPO)
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--','trackb'],cwd=REPO,text=True).splitlines()
    for name in untracked:
        diff=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',name],cwd=REPO,capture_output=True)
        if diff.returncode not in (0,1):raise RuntimeError(diff.stderr.decode())
        patch+=diff.stdout
    (out/'code.patch').write_bytes(patch)
    with zipfile.ZipFile(out/'trackb-code.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in files:z.write(REPO/n,n)
    (out/'receipt.json').write_text(json.dumps({'baseline':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'files':{n:hashlib.sha256((REPO/n).read_bytes()).hexdigest() for n in files},
        'zip_sha256':hashlib.sha256((out/'trackb-code.zip').read_bytes()).hexdigest(),
        'patch_sha256':hashlib.sha256(patch).hexdigest()},indent=2))
    print(f'{len(files)} source files: {out}')
if __name__=='__main__':main()
