#!/usr/bin/env bash
# Daily ZFS backup with snapshot rotation
# Keeps 7 daily, 4 weekly (Sundays), 12 monthly (1st) snapshots on SOURCE
# Replicates recursively to BACKUP pool, incrementally when possible.

set -euo pipefail

BACKUP_POOL="backup"
DATE=$(date +%Y%m%d)
TAG="daily-${DATE}"

# check if the backup pool is available
if ! zpool list -H -o name 2>/dev/null | grep -qx "$BACKUP_POOL"; then
  echo "Backup pool $BACKUP_POOL is not available; exiting."
  exit 0
fi

# --- Pools to protect (explicit includes) ---
# Include rpool, data, and bpool; exclude BACKUP_POOL only.
CANDIDATE_POOLS=("rpool" "data" "bpool")

# Filter to only those that exist/imported and are not BACKUP_POOL
POOLS=()
for P in "${CANDIDATE_POOLS[@]}"; do
  if [ "$P" != "$BACKUP_POOL" ] && zpool list -H -o name 2>/dev/null | grep -qx "$P"; then
    POOLS+=("$P")
  fi
done

if [ ${#POOLS[@]} -eq 0 ]; then
  echo "No source pools found; exiting."
  exit 0
fi

echo "==> Backing up pools: ${POOLS[*]}  (target: $BACKUP_POOL) at $(date)"

# ---------------------------
# 1) Create recursive snapshots (once per pool)
# ---------------------------
for POOL in "${POOLS[@]}"; do
  SNAP="${POOL}@${TAG}"
  echo "  -> snapshot $SNAP (recursive)"
  if ! zfs list -t snapshot -o name | grep -q "^${SNAP}$"; then
    zfs snapshot -r "$SNAP"
  else
    echo "     (already exists)"
  fi
done

# ---------------------------
# 2) Replicate each dataset to backup (incremental when possible)
# ---------------------------
for POOL in "${POOLS[@]}"; do
  echo "==> Replicating pool: $POOL"

  while IFS= read -r DS; do
    SRC_SNAP="${DS}@${TAG}"
    DST_FS="${BACKUP_POOL}/${DS}"

    # Ensure source snapshot exists (created via pool-level recursive snapshot)
    if ! zfs list -t snapshot -o name 2>/dev/null | grep -q "^${SRC_SNAP}$"; then
      echo "  -> skip (no snapshot for dataset): $SRC_SNAP"
      continue
    fi

    # Ensure destination hierarchy exists
    if ! zfs list -H -o name "$DST_FS" >/dev/null 2>&1; then
      echo "  -> create dest: $DST_FS"
      zfs create -p "$DST_FS"
    fi

    # Find previous source snapshot for this dataset (latest older daily-YYYYmmdd)
    PREV=$(zfs list -t snapshot -o name -s creation 2>/dev/null \
           | awk -v ds="$DS" -v cur="@${TAG}" '
               index($0, ds"@daily-")==1 && $0 !~ cur { last=$0 } END{ if (last) print last }')

    echo -n "  -> send/recv: $SRC_SNAP -> $DST_FS  "
    if [ -n "${PREV:-}" ]; then
      echo "(incremental from $PREV)"
      zfs send -R -i "$PREV" "$SRC_SNAP" | zfs receive -uF "$DST_FS"
    else
      echo "(full)"
      zfs send -R "$SRC_SNAP" | zfs receive -uF "$DST_FS"
    fi
  done < <(zfs list -H -o name -r "$POOL")
done

echo "==> Backup completed at $(date)"

# ------------------------
# 3) Snapshot pruning policy (on SOURCE)
#    - keep newest 7 dailies
#    - keep 4 weekly (those whose date is Sunday)
#    - keep 12 monthly (those on day 01)
# ------------------------
echo "==> Pruning old snapshots on source..."

is_sunday() {
  local ymd="$1"   # YYYYMMDD
  [ "$(date -d "${ymd}" +%u)" = "7" ] 2>/dev/null
}

for POOL in "${POOLS[@]}"; do
  while IFS= read -r DS; do
    # All daily snapshots for this dataset, sorted old->new
    MAPFILE -t ALL < <(zfs list -t snapshot -o name -s creation 2>/dev/null \
                       | grep -E "^${DS}@daily-[0-9]{8}$" || true)
    [ ${#ALL[@]} -eq 0 ] && continue

    # Keep newest 7 daily
    DAILY_KEEP=7
    if [ ${#ALL[@]} -le $DAILY_KEEP ]; then
      KEEP_DAILY=("${ALL[@]}")
    else
      KEEP_DAILY=("${ALL[@]: -$DAILY_KEEP}")
    fi

    # Keep up to 4 weekly (Sundays), scanning newest->oldest
    KEEP_WEEKLY=()
    count=0
    for ((i=${#ALL[@]}-1; i>=0; i--)); do
      ymd="${ALL[$i]##*@daily-}"
      if is_sunday "$ymd"; then
        KEEP_WEEKLY+=("${ALL[$i]}")
        ((count++))
        [ $count -ge 4 ] && break
      fi
    done

    # Keep up to 12 monthly (1st of month), scanning newest->oldest
    KEEP_MONTHLY=()
    count=0
    last_month=""
    for ((i=${#ALL[@]}-1; i>=0; i--)); do
      snap="${ALL[$i]}"
      ymd="${snap##*@daily-}"     # YYYYMMDD
      day="${ymd:6:2}"
      month="${ymd:0:6}"          # YYYYMM
      if [ "$day" = "01" ] && [ "$month" != "$last_month" ]; then
        KEEP_MONTHLY+=("$snap")
        last_month="$month"
        ((count++))
        [ $count -ge 12 ] && break
      fi
    done

    # Combine keep list (unique)
    KEEP_SET=$(printf "%s\n" "${KEEP_DAILY[@]}" "${KEEP_WEEKLY[@]}" "${KEEP_MONTHLY[@]}" | sort -u)

    # Destroy others
    for snap in "${ALL[@]}"; do
      if ! printf "%s\n" "$KEEP_SET" | grep -qx "$snap"; then
        echo "  -> Destroying old snapshot: $snap"
        zfs destroy -r "$snap"
      fi
    done
  done < <(zfs list -H -o name -r "$POOL")
done

echo "==> Pruning finished at $(date)"

# tail -f /var/log/zfs-backup.log
