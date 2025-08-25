# scp -r /Users/weichenzhang/Library/Containers/com.tencent.WeWorkMac/Data/Documents/Profiles/601920D267682C5A320232CFB0B98E28/Caches/Files/2025-08/56dc696de8281628c8016fd63583dbbd/hps_res_demo.zip debug-H20-0:/picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/


# scp -r *.ckpt debug-H20-0:/picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/

#   --delete \

SOURCE_DIR=/home/cevin/Meitu/
TARGET_DIR=/picassox/vepfs-mtlab-train-base-new/human-body/weichen.zhang/

rsync -avP \
  --exclude-from=.rsyncignore \
  --partial \
  --inplace \
  -e "ssh -o Compression=yes" \
  $SOURCE_DIR  debug-H20-0:$TARGET_DIR
