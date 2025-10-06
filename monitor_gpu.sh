#!/bin/bash

# GPU and CPU Monitoring Script
# Usage: ./monitor_gpu.sh [interval_seconds]

INTERVAL=${1:-2}  # Default 2 seconds

# Pick a Python interpreter for fallback metrics
if command -v python3 >/dev/null 2>&1; then
    PY_BIN=python3
elif [ -x "/home/myenv/bin/python" ]; then
    PY_BIN=/home/myenv/bin/python
else
    PY_BIN=""
fi

# Detect if NVML/nvidia-smi is usable
if nvidia-smi -L >/dev/null 2>&1; then
    NVML_OK=1
else
    NVML_OK=0
fi

echo "🚀 GPU/CPU Monitor - Press Ctrl+C to stop"
echo "Interval: ${INTERVAL}s"
echo "=========================================="

while true; do
    # Clear screen and show timestamp
    clear
    echo "🕐 $(date '+%H:%M:%S') - GPU/CPU Monitor"
    echo "=========================================="
    
    # GPU Memory Usage
    if [ "$NVML_OK" -eq 1 ]; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
        GPU_USED=$(echo "$GPU_MEM" | cut -d',' -f1 | tr -d ' ')
        GPU_TOTAL=$(echo "$GPU_MEM" | cut -d',' -f2 | tr -d ' ')
        if [[ "$GPU_USED" =~ ^[0-9]+$ && "$GPU_TOTAL" =~ ^[0-9]+$ && "$GPU_TOTAL" -gt 0 ]]; then
            GPU_PERCENT=$(echo "scale=1; $GPU_USED * 100 / $GPU_TOTAL" | bc)
            echo "💾 GPU Memory: ${GPU_PERCENT}% (${GPU_USED}MB / ${GPU_TOTAL}MB)"
        else
            echo "💾 GPU Memory: unavailable (NVML parse error)"
        fi
    else
        # Python fallback using torch if available
        if [ -n "$PY_BIN" ]; then
            PY_OUT=$($PY_BIN - <<'PY'
import sys
try:
    import torch
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        print(int(used/1e6), int(total/1e6))  # MB
    else:
        print("NA NA")
except Exception:
    print("NA NA")
PY
)
            USED_MB=$(echo "$PY_OUT" | awk '{print $1}')
            TOTAL_MB=$(echo "$PY_OUT" | awk '{print $2}')
            if [[ "$USED_MB" =~ ^[0-9]+$ && "$TOTAL_MB" =~ ^[0-9]+$ && "$TOTAL_MB" -gt 0 ]]; then
                PCT=$(echo "scale=1; $USED_MB * 100 / $TOTAL_MB" | bc)
                echo "💾 GPU Memory: ${PCT}% (${USED_MB}MB / ${TOTAL_MB}MB)"
            else
                echo "💾 GPU Memory: unavailable (NVML down, Python fallback failed)"
            fi
        else
            echo "💾 GPU Memory: unavailable (NVML down, no Python found)"
        fi
    fi
    
    # GPU Utilization
    if [ "$NVML_OK" -eq 1 ]; then
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')
        if [[ "$GPU_UTIL" =~ ^[0-9]+$ ]]; then
            echo "⚡ GPU Utilization: ${GPU_UTIL}%"
        else
            echo "⚡ GPU Utilization: unavailable (NVML parse error)"
        fi
    else
        echo "⚡ GPU Utilization: unavailable (NVML down)"
    fi
    
    # CPU Usage
    CPU_USAGE=$(top -bn1 | awk -F',' '/Cpu\(s\)/{gsub(/.*: /,""); gsub(/ id.*/,""); print 100-$4}')
    echo "🖥️  CPU Usage: ${CPU_USAGE}%"
    
    # Active Python processes
    PYTHON_COUNT=$(ps aux | grep python | grep -v grep | wc -l)
    echo "🐍 Python Processes: ${PYTHON_COUNT}"
    
    # Memory usage of main process
    MAIN_PID=$(ps aux | grep compact_model_comparison | grep -v grep | head -1 | awk '{print $2}')
    if [ ! -z "$MAIN_PID" ]; then
        MAIN_MEM=$(ps -p $MAIN_PID -o %mem --no-headers | tr -d ' ')
        echo "📊 Main Process Memory: ${MAIN_MEM}%"
    fi
    
    echo "=========================================="
    echo "Press Ctrl+C to stop monitoring"
    
    sleep $INTERVAL
done


# GPU Memory Usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# GPU Utilization (should be higher with more workers)
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits

# CPU Usage
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//'

# One-liner for all metrics
watch -n 1 'echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr "," " " | awk "{print \$1\"% util, \"\$2\"MB mem\"}") | CPU: $(top -bn1 | grep "Cpu(s)" | awk "{print \$2}" | sed "s/%us,//")%"'