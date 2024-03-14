while true; do
    current_time=$(date "+%Y-%m-%d %H:%M:%S")
    used_ram=$(free -m | awk 'NR==2 {print $3}')
    echo "$current_time $used_ram"
    sleep 1
done
