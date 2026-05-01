import numpy as np
from datasets import Dataset

ds = Dataset.load_from_disk("datasets/sft_chat_v2/shard-000")

def stats(row):
    labels = np.asarray(row["labels"], dtype=np.int64)
    active = labels != -100
    if not active.any():
        return None
    first = int(active.argmax())
    frac = float(active.mean())
    count = int(active.sum())
    return first, frac, count

firsts = []
fracs = []
counts = []
empty = 0

for i in range(len(ds)):
    s = stats(ds[i])
    if s is None:
        empty += 1
        continue
    f, fr, c = s
    firsts.append(f)
    fracs.append(fr)
    counts.append(c)

print(f"samples: {len(ds)}")
print(f"empty (no supervised tokens): {empty} ({empty/len(ds)*100:.2f}%)")
if firsts:
    print(f"first supervised token index: p50={int(np.median(firsts))} p90={int(np.quantile(firsts,0.9))} p99={int(np.quantile(firsts,0.99))}")
    print(f"supervised fraction: p50={np.median(fracs):.3f} p90={np.quantile(fracs,0.9):.3f}")
    print(f"supervised tokens per 1024: p50={int(np.median(counts))} p90={int(np.quantile(counts,0.9))}")
