du -h -d 1 / | sort -h

du -h -d 1 /tmp | sort -h
find /tmp -xdev -user "$USER" -mtime +3 -delete

du -h -d 1 /opt | sort -h

apt remove google-chrome-stable