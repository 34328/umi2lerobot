#!/bin/bash
# FastUMI 批量转换脚本
# 用于批量将 FastUMI HDF5 数据集转换为 LeRobot 格式
# 
# 使用方法:
#   chmod +x convert_fastumi_all.sh
#   ./convert_fastumi_all.sh
#
# 也可以后台运行:
#   nohup ./convert_fastumi_all.sh > convert_log.txt 2>&1 &

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/src/hdf5/fastumi_hdf52lerobot.py"
CONDA_ENV="umi2lerobot"

# 激活 conda 环境
echo -e "${YELLOW}激活 conda 环境: ${CONDA_ENV}${NC}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# ============================================================================
# 任务定义 (subtask -> text 映射)
# ============================================================================
# 格式: "subtask|text描述"
# 
# 已完成的任务 (已注释):
# "clean_table|Clean the table"
# "close_ricecooker|Close the rice cooker"
# "cover_beef|Cover the beef with a lid"
# "fold_towel|Fold the towel"
# "hotdog_in_rice_cooker|Put the hotdog in the rice cooker"
# "hotdog_in_roaster|Put the hotdog in the roaster"
# "open_container|Open the container"
# "open_drawer|Open the drawer"

# 待转换的任务列表
TASKS=(
    "open_ricecooker|Open the rice cooker"
    "open_roaster|Open the roaster"
    "open_suitcase|Open the suitcase"
    "pick_bear|Pick up the bear and put it in the box"
    "pick_bread|Pick up the bread and put it on the plate"
    "pick_cup|Pick up the cup and place it on the mat"
    "pick_lid|Pick up the lid and place it in box"
    "pick_pen|Pick up the pen and place it in the pen holder"
    "place_plate|Pick up the pen and place it in the pen holder"
    "place_pot|Place the pot on the stove"
    "pour_coke|Pour the coke into the cpu"
    "rearrange_coke|Put the coke bottle on the second level of the cabinet"
    "sweep_trash|Sweep the trash into the dustpan"
    "unplug_charger|Unplug the charger"
)

# ============================================================================
# 主转换逻辑
# ============================================================================

# 统计
TOTAL=${#TASKS[@]}
CURRENT=0
SUCCESS=0
FAILED=0
FAILED_TASKS=()

echo ""
echo "=============================================="
echo "  FastUMI 批量转换脚本"
echo "  共 ${TOTAL} 个任务待转换"
echo "=============================================="
echo ""

START_TIME=$(date +%s)

for task_entry in "${TASKS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # 解析 subtask 和 text
    IFS='|' read -r subtask text <<< "${task_entry}"
    
    echo ""
    echo -e "${YELLOW}=============================================="
    echo -e "  [$CURRENT/$TOTAL] 正在转换: ${subtask}"
    echo -e "  任务描述: ${text}"
    echo -e "==============================================${NC}"
    echo ""
    
    # 执行转换
    python "${PYTHON_SCRIPT}" \
        --subtask "${subtask}" \
        --text "${text}" \
        --resume
    
    # 检查返回值
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ${subtask} 转换成功!${NC}"
        SUCCESS=$((SUCCESS + 1))
    else
        echo -e "${RED}✗ ${subtask} 转换失败!${NC}"
        FAILED=$((FAILED + 1))
        FAILED_TASKS+=("${subtask}")
    fi
    
    echo ""
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=============================================="
echo "  转换完成!"
echo "=============================================="
echo "  总任务数: ${TOTAL}"
echo -e "  ${GREEN}成功: ${SUCCESS}${NC}"
echo -e "  ${RED}失败: ${FAILED}${NC}"
echo "  总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

if [ ${FAILED} -gt 0 ]; then
    echo -e "${RED}失败的任务:${NC}"
    for task in "${FAILED_TASKS[@]}"; do
        echo "  - ${task}"
    done
    echo ""
fi

echo "=============================================="
