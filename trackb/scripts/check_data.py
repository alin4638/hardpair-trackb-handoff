"""Phase 2 checks: per-class counts, manifest (stand-in or Track A's), local paths, and the leakage guard.

    python scripts/check_data.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tb import data  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


print("== per-class counts vs Zenodo 1214456")
tiles = {ds: data.list_tiles(ds) for ds in data.DATASETS}
for ds, t in tiles.items():
    c = data.check_counts(ds, t)
    print(f"-- {data.DATASETS[ds]} (total {c.local.sum()})\n{c.T.to_string()}")
    check(f"{ds} counts match", bool(c.ok.all()))
check("7K total = 7,180", len(tiles["7k"]) == 7180)
check("100K total = 100,000", len(tiles["100k"]) == 100000)
check("filenames unique within each dataset", all(t.fname.is_unique for t in tiles.values()))
check("no filename shared between 100K and 7K", not set(tiles["100k"].fname) & set(tiles["7k"].fname))

print(f"\n== manifest ({data.manifest_source().name})")
m = data.load_manifest()
print(m.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
check("train = 2,000/class", (m[m.split == "train"].label.value_counts() == 2000).all() and m.split.eq("train").sum() == 18000)
pe = m[m.split != "train"].groupby(["label", "split"]).size().unstack()
check("pool/eval = 50/50 per class (diff <= 1)", bool(((pe["pool"] - pe["eval"]).abs() <= 1).all()), f"pool {pe['pool'].sum()} eval {pe['eval'].sum()}")
check("every manifest path resolves locally", all(Path(p).exists() for p in m.local_path))
check("train tiles come from 100K only, pool/eval from 7K only",
      m[m.split == "train"].path.str.contains("HE-100K").all() and m[m.split != "train"].path.str.contains("HE-7K").all())
if data.manifest_source() == data.STANDIN_MANIFEST:
    m2 = data.build_standin(ROOT / "data" / "_standin_rebuild.csv")
    check("stand-in rebuild is deterministic", m2[["tile_id", "path", "split"]].equals(m[["tile_id", "path", "split"]]))
    (ROOT / "data" / "_standin_rebuild.csv").unlink()

print("\n== leakage guard")
rows = data.training_rows(m)
check("training_rows: train only", set(rows.split) == {"train"} and len(rows) == 18000)
rows_x = data.training_rows(m, include_extra=True)
check("training_rows + extra NCT: no 7K tile", not rows_x.path.str.contains("HE-7K").any(), f"({len(rows_x)} rows, {rows_x.split.eq('train_extra').sum()} extra)")

ev, po = m[m.split == "eval"], m[m.split == "pool"]


def raises(fn):
    try:
        fn()
    except data.LeakError:
        return True
    return False


check("eval tile injected -> LeakError", raises(lambda: data.assert_no_leak(pd.concat([rows, ev.head(1)]), m)))
check("pool tile injected (not allowed) -> LeakError", raises(lambda: data.assert_no_leak(pd.concat([rows, po.head(1)]), m)))
renamed = ev.head(1).assign(tile_id="nct_sneaky")
check("eval tile under a fake id -> LeakError (filename match)", raises(lambda: data.assert_no_leak(pd.concat([rows, renamed]), m)))
log = ROOT / "outputs" / "_check_corrections.csv"
corr = data.training_rows(m, pool_ids=po.tile_id.head(5).tolist(), log_path=log)
check("allowed pool corrections pass and are logged", len(corr) == 18005 and len(pd.read_csv(log)) == 5)
log.unlink()
check("eval ids passed as 'corrections' -> LeakError",
      raises(lambda: data.training_rows(m, pool_ids=ev.tile_id.head(2).tolist(), log_path=log)))

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
