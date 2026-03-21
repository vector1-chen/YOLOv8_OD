#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v roscore >/dev/null 2>&1; then
  echo "[ERROR] ROS 未安装或未在 PATH 中"
  exit 1
fi

if [[ -z "$ROS_DISTRO" ]]; then
  echo "[INFO] 未检测到 ROS 环境，请先在当前终端执行: sros"
  exit 1
fi

echo "[INFO] 启动 Gazebo..."
roslaunch "$ROOT_DIR/tool/model_capture/capture_dumpster.launch" > /tmp/dumpster_gazebo.log 2>&1 &
GAZEBO_PID=$!

cleanup() {
  echo "[INFO] 停止 Gazebo..."
  kill "$GAZEBO_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

sleep 6

echo "[INFO] 开始采集..."
python3 "$ROOT_DIR/tool/model_capture/capture_dumpster_dataset.py" "$@"

echo "[INFO] 采集完成"
