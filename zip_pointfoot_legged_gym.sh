#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
用法: zip_pointfoot_legged_gym.sh [-o 输出zip文件名] [-d 输出目录]
- 默认输出名: pointfoot-legged-gym_YYYYmmdd_HHMMSS.zip
- 默认输出目录: 脚本所在目录
- 始终排除: pointfoot-legged-gym/logs/*

示例:
  bash zip_pointfoot_legged_gym.sh
  bash zip_pointfoot_legged_gym.sh -o pf.zip -d /tmp
EOF
}

ZIP_BASENAME=""
OUT_DIR=""

while getopts ":o:d:h" opt; do
  case $opt in
    o) ZIP_BASENAME="$OPTARG" ;;
    d) OUT_DIR="$OPTARG" ;;
    h) show_help; exit 0 ;;
    \?) echo "非法参数: -$OPTARG" >&2; show_help; exit 1 ;;
    :) echo "选项 -$OPTARG 需要参数" >&2; show_help; exit 1 ;;
  esac
done

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$BASE_DIR/pointfoot-legged-gym"
if [[ ! -d "$TARGET_DIR" ]]; then
  echo "未找到目录: $TARGET_DIR" >&2
  exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
  echo "未找到zip命令，请先安装: sudo apt-get install -y zip" >&2
  exit 1
fi

timestamp=$(date +%Y%m%d_%H%M%S)
if [[ -z "$ZIP_BASENAME" ]]; then
  ZIP_BASENAME="pointfoot-legged-gym_${timestamp}.zip"
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$BASE_DIR"
fi

mkdir -p "$OUT_DIR"
ZIP_PATH="$OUT_DIR/$ZIP_BASENAME"

# 若存在同名文件则先删除
if [[ -f "$ZIP_PATH" ]]; then
  rm -f "$ZIP_PATH"
fi

cd "$BASE_DIR"
echo "正在压缩 $TARGET_DIR -> $ZIP_PATH (排除 logs/)..."
zip -r "$ZIP_PATH" "pointfoot-legged-gym" -x "pointfoot-legged-gym/logs/*"

echo "完成: $ZIP_PATH"
