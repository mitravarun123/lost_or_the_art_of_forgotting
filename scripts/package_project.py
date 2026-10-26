"""Create the source ZIP accepted by the master notebook; exclude past results/data."""
from pathlib import Path
import hashlib
import json
import zipfile


def package(root, destination):
    root=Path(root).resolve();destination=Path(destination).resolve()
    skip={'__pycache__','.pytest_cache','.venv','.git','datasets','results','logs','__MACOSX'}
    files=[]
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(root.rglob('*')):
            if not file.is_file():continue
            rel=file.relative_to(root)
            if skip.intersection(rel.parts) or file.suffix.lower() in {'.zip','.pdf','.aux','.log','.out','.pyc'}:continue
            archive.write(file,str(Path(root.name)/rel));files.append(str(rel))
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    return dict(archive=str(destination),sha256=digest,source_files=len(files),bytes=destination.stat().st_size)

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    print(json.dumps(package(root,root.parent/'lost-or-inaccessible.zip'),indent=2))
