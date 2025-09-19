import os
import cv2
import json

check_folder = '/picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/training-images/'
ds_peak_folder = '/picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/CameraHMR/data/training-images-peak/'

ds_size = {}
for root, dirs, files in os.walk(check_folder):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            ds_folder = root.replace(check_folder, "").split('/')[0]
            img = cv2.imread(os.path.join(root, file))
            if ds_folder not in ds_size:
                ds_size[ds_folder] = set([img.shape])
                cv2.imwrite(os.path.join(ds_peak_folder, file), img)
            else:
                ds_size[ds_folder].add(img.shape)

for ds_folder, size in ds_size.items():
    print(ds_folder, size)

# save all ds_size to a json file
with open('ds_size.json', 'w') as f:
    json.dump(ds_size, f)