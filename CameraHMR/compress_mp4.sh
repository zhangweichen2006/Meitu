IN="/Users/weichenzhang/Downloads/Download_overlay.mp4"
OUT="/Users/weichenzhang/Downloads/Download_overlay_<=5MB.mp4"
DUR=$(ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$IN")
BR_K=$(python3 - <<PY
dur=float("$DUR"); target_bits=int(4.8*1024*1024*8*0.97)  # 3% mux overhead
print(max(300, int(target_bits/dur/1000)))
PY
)
ffmpeg -y -i "$IN" -an -c:v libx264 -b:v ${BR_K}k -maxrate ${BR_K}k -bufsize $((2*BR_K))k -preset slow -pass 1 -f mp4 /dev/null
ffmpeg -i "$IN" -an -c:v libx264 -b:v ${BR_K}k -maxrate ${BR_K}k -bufsize $((2*BR_K))k -preset slow -movflags +faststart -pass 2 "$OUT"
rm -f ffmpeg2pass-0.log*