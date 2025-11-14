from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter
import glob, os

src = "logs/sparknet-70m-v2"
dst = "logs/sparknet-70m-v2-merged"
os.makedirs(dst, exist_ok=True)

writer = SummaryWriter(dst)
for f in sorted(glob.glob(os.path.join(src, "events.out.tfevents.*"))):
    ea = event_accumulator.EventAccumulator(f)
    ea.Reload()
    for tag in ea.Tags().get("scalars", []):
        for ev in ea.Scalars(tag):
            writer.add_scalar(tag, ev.value, ev.step)
writer.close()
print("✅ Merged TensorBoard logs written to:", dst)
