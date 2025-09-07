# === 需要按需修改的变量 ===
export DISK=/dev/nvme0n1         # 目标磁盘
export HOSTNAME=myhost
export USERNAME=cevin
export TIMEZONE=Australia/Sydney  # 时区
export LOCALE=en_AU.UTF-8         # 语言
export ROOT_SIZE_GB=200           # rpool 大小（根）
export BPOOL_SIZE_GB=2            # bpool 大小（/boot, ZFS 方案）
export EFI_SIZE_MB=1024           # EFI 分区
export SWAP_SIZE_GB=32             # 如不需要可设为0
# 发行版代号（Ubuntu 24.04 是 noble）
export SUITE=noble

set -euxo pipefail

# 卸载/导出以免占用
swapoff -a || true
zpool export -a || true

# 清盘并分区
sgdisk --zap-all $DISK

# 对齐用 1MiB 开始
# p1 EFI
sgdisk -n1:1MiB:+${EFI_SIZE_MB}MiB  -t1:EF00 -c1:"EFI System" $DISK
# p2 bpool (ZFS /boot)
sgdisk -n2:0:+${BPOOL_SIZE_GB}GiB   -t2:BF00 -c2:"bpool"      $DISK
# p3 rpool (ZFS /)
sgdisk -n3:0:+${ROOT_SIZE_GB}GiB    -t3:BF00 -c3:"rpool"      $DISK
# p4 datapool (ZFS /home & data) —— 占余下所有空间（减去 swap）
if [ "$SWAP_SIZE_GB" -gt 0 ]; then
  sgdisk -n4:0:-${SWAP_SIZE_GB}GiB  -t4:BF00 -c4:"datapool"   $DISK
  sgdisk -n5:0:0                     -t5:8200 -c5:"swap"       $DISK
else
  sgdisk -n4:0:0                     -t4:BF00 -c4:"datapool"   $DISK
fi

partprobe $DISK
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT $DISK

# EFI
mkfs.vfat -F32 -n EFI ${DISK}p1

########### Step 2 ###########
# 检测是否支持 ZFS 池兼容属性（给 bpool 用）
if zpool create -n -o compatibility=grub2 testpool ${DISK}p2 2>/dev/null; then
  BPOOL_COMPAT="-o compatibility=grub2"
else
  BPOOL_COMPAT=""   # 回退：后面会改用 ext4 /boot
fi

# 若支持 ZFS /boot（bpool）
if [ -n "$BPOOL_COMPAT" ]; then
  zpool create -f -o ashift=12 -d \
    -o cachefile=/etc/zfs/zpool.cache \
    $BPOOL_COMPAT \
    -O devices=off -O mountpoint=none -O compression=zstd -O atime=off \
    bpool ${DISK}p2
fi

# rpool（根）
zpool create -f -o ashift=12 \
  -O acltype=posixacl -O xattr=sa -O dnodesize=auto \
  -O normalization=formD -O relatime=on \
  -O compression=zstd -O mountpoint=none \
  rpool ${DISK}p3

# datapool（home & data）
zpool create -f -o ashift=12 \
  -O acltype=posixacl -O xattr=sa -O relatime=on \
  -O compression=zstd -O mountpoint=none \
  datapool ${DISK}p4

# 如有 swap 分区
if [ "$SWAP_SIZE_GB" -gt 0 ]; then
  mkswap -L swap ${DISK}p5
fi
