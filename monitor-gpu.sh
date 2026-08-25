#!/usr/bin/env bash
set -euo pipefail

interval=${1:-1}

if [[ ! $interval =~ ^([1-9][0-9]*([.][0-9]+)?|0[.][0-9]*[1-9][0-9]*)$ ]]; then
    echo "Usage: $0 [refresh-seconds]" >&2
    exit 2
fi

for command in docker nvidia-smi; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command not found: $command" >&2
        exit 1
    fi
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
compose=(docker compose -f "$script_dir/docker-compose.yml" --profile upstream)
service=upstream-candidate

begin_frame() {
    printf '\033[H'
    printf 'Qwen3.8 GPU monitor - %s - refresh: %ss\n\n' "$timestamp" "$interval"
}

end_frame() {
    printf '\033[J'
}

cleanup() {
    printf '\033[?25h\n'
}

trap cleanup EXIT
trap 'exit 0' INT TERM
printf '\033[?25l'

while true; do
    timestamp=$(date '+%F %T')

    container_id=$("${compose[@]}" ps -q "$service" 2>/dev/null || true)
    if [[ -z $container_id ]]; then
        begin_frame
        printf 'Service %s does not exist. Start it with %s/run.sh.\n' "$service" "$script_dir"
        end_frame
        sleep "$interval"
        continue
    fi

    container_state=$(docker inspect --format '{{.State.Status}}' "$container_id" 2>/dev/null || true)
    if [[ $container_state != running ]]; then
        begin_frame
        printf 'Service: %s  Container: %.12s  State: %s\n' "$service" "$container_id" "${container_state:-unknown}"
        end_frame
        sleep "$interval"
        continue
    fi

    container_pids=$(docker top "$container_id" -eo pid= 2>/dev/null | awk '$1 ~ /^[0-9]+$/ { print $1 }' | tr '\n' ' ' || true)
    app_rows=$(docker exec "$container_id" nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)
    if [[ -z $app_rows ]]; then
        app_rows=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)
    fi
    matching_apps=$(awk -F ', *' -v pids="$container_pids" '
        BEGIN {
            count = split(pids, values, " ")
            for (i = 1; i <= count; i++) {
                if (values[i] != "") wanted[values[i]] = 1
            }
        }
        $2 in wanted { print }
    ' <<< "$app_rows")
    gpu_rows=$(nvidia-smi --query-gpu=uuid,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null || true)
    if [[ -z $gpu_rows ]]; then
        begin_frame
        printf 'Service: %s  Container: %.12s  State: %s\n\n' "$service" "$container_id" "$container_state"
        echo 'nvidia-smi could not read GPU metrics.'
        end_frame
        exit 1
    fi

    relevant_uuids=$(docker exec "$container_id" nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | awk '{$1=$1; print}' || true)
    if [[ -z $relevant_uuids ]]; then
        relevant_uuids=$(awk -F ', *' 'NF { print $1 }' <<< "$matching_apps" | sort -u)
    fi
    if [[ -z $relevant_uuids ]]; then
        relevant_uuids=$(awk -F ', *' 'NF { print $1 }' <<< "$gpu_rows")
    fi

    begin_frame
    printf 'Service: %s  Container: %.12s  State: %s\n\n' "$service" "$container_id" "$container_state"
    printf '%-3s %-24s %7s %7s %18s %7s %17s\n' GPU NAME UTIL MEM-UTIL VRAM TEMP POWER
    awk -F ', *' -v uuids="$relevant_uuids" '
        BEGIN {
            count = split(uuids, values, /[[:space:]]+/)
            for (i = 1; i <= count; i++) wanted[values[i]] = 1
        }
        $1 in wanted {
            printf "%-3s %-24.24s %6s%% %6s%% %7s / %-7s MiB %5s C %7s / %-7s W\n", $2, $3, $4, $5, $6, $7, $8, $9, $10
        }
    ' <<< "$gpu_rows"

    if [[ -n $matching_apps ]]; then
        printf '\nContainer GPU processes:\n'
        printf '%-8s %-24s %12s\n' PID PROCESS VRAM
        awk -F ', *' '{ printf "%-8s %-24.24s %9s MiB\n", $2, $3, $4 }' <<< "$matching_apps"
    else
        printf '\nPer-process GPU accounting is unavailable. GPU totals above remain valid.\n'
        echo 'This is expected when Docker and nvidia-smi use different PID namespaces.'
    fi
    end_frame

    sleep "$interval"
done
