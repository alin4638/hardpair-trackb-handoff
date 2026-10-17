"""B contract and actual dataset checks (requires restored data)."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tb.common import load_config
from tb.pipeline import prepare
from tb.data import load_manifest,training_rows,EXPECTED_COUNTS,DATASETS
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/laptop_4080.yaml');args=p.parse_args()
c=load_config(args.config);print(prepare(c,True));m=load_manifest(True,c)
assert len(training_rows(m))==16200
for ds,classes in EXPECTED_COUNTS.items():
    for label,count in classes.items():
        got=len(list((Path(c['data_root'])/DATASETS[ds]/label).glob('*.tif')))
        if got!=count:raise ValueError(f'{ds}/{label}: found {got}, expected {count}')
print('PASS: frozen B contracts, local paths, 16200 fit tiles and complete dataset counts')
