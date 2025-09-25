import os, re

root = '/home/cevin/Meitu/CameraHMR/data/training-images/20221011_1_250_batch01hand_closeup_suburb_a_6fps'

for rt, _, files in os.walk(root):
    for f in files:
        if f.lower().endswith(('.jpg', '.png')):
            m = re.search(r'(\d+)(?=\.[^.]+$)', f)  # digits before extension
            if not m:
                continue
            idx = int(m.group(1))
            fp = os.path.join(rt, f)
            if idx % 5 != 0:
                os.remove(fp)
                print(f"Removed {fp}")
            else:
                print(f"Kept {fp}")