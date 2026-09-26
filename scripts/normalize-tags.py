#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 TAGS.md 词表归一化所有文章的 frontmatter tags。

用法: python3 scripts/normalize-tags.py [--apply]
不带 --apply 时只打印 dry-run 报告。
"""
import re, glob, sys, collections

APPLY = "--apply" in sys.argv

# 旧标签 -> 规范名（来自 TAGS.md 同义词对照）
MAP = {
    # 记忆
    "智能体记忆": "记忆", "经验记忆": "记忆", "记忆机制": "记忆",
    "记忆系统": "记忆", "长期记忆": "记忆", "隐式关联": "记忆", "检索假设": "记忆",
    "记梦": "记忆",
    # 推理与规划
    "智能体推理": "推理与规划", "智能体决策": "推理与规划", "多步推理": "推理与规划",
    "决策判断": "推理与规划", "决策分叉": "推理与规划", "智能体规划": "推理与规划",
    "规划": "推理与规划", "长程规划": "推理与规划", "长程推理": "推理与规划",
    "任务自适应": "推理与规划", "部分可观测": "推理与规划", "图推理": "推理与规划",
    # 认知架构
    "智能体认知": "认知架构", "智能体认知架构": "认知架构", "智能体架构": "认知架构",
    "supervisor-worker架构": "认知架构", "感知": "认知架构",
    # 具身智能
    "具身智能体": "具身智能", "语言控制": "具身智能",
    # 评估与基准
    "智能体评测": "评估与基准", "认知基准": "评估与基准",
    "选择性预测": "评估与基准", "置信度估计": "评估与基准",
    "可审计性": "评估与基准", "品味评估": "评估与基准", "视频生成评估": "评估与基准",
    # 自我改进
    "自进化": "自我改进", "递归自改进": "自我改进", "智能体自改进": "自我改进",
    "Harness进化": "自我改进", "递归数据引擎": "自我改进",
    # 技能学习
    "技能演化": "技能学习",
    # 长程任务
    "长程任务": "长程任务", "生命周期任务": "长程任务", "持久智能体": "长程任务",
    # 上下文管理
    "上下文管理": "上下文管理", "个人上下文": "上下文管理",
    "状态修正": "上下文管理", "任务状态污染": "上下文管理", "读取时裁决": "上下文管理",
    # 知识蒸馏
    "轨迹蒸馏": "知识蒸馏",
    # 策略优化
    "GRPO": "策略优化", "可验证奖励": "策略优化", "信用分配": "策略优化",
    # 机器学习
    "正则化": "机器学习", "泛化": "机器学习", "测试时优化": "机器学习",
    "LoRA适配": "机器学习",
    # 矩阵方法
    "PCA": "矩阵方法", "核方法": "矩阵方法", "矩阵分解": "矩阵方法",
    # 推荐系统
    "协同过滤": "推荐系统",
    # 游戏智能体
    "游戏开发": "游戏智能体",
    # 生活
    "日记": "生活", "随想": "生活", "期末": "生活", "重生": "生活",
    "安全": "生活", "小满": "生活", "南京": "生活",
}

# 日记类文章单独处理：category=日记，tags 至少含「生活」和「日记」
DIARY_KEEP = {"日记"}

KEEP = {"世界模型", "多智能体", "强化学习", "推理与规划", "记忆", "认知架构",
        "工具使用", "具身智能", "评估与基准", "游戏智能体", "经济模拟",
        "机器学习", "Web开发", "生活", "日记",
        "元认知", "自我改进", "技能学习", "长程任务", "上下文管理",
        "知识蒸馏", "空间智能", "物理推理", "视频生成", "策略优化",
        "推荐系统", "矩阵方法", "Web评测", "摄影", "玄武湖", "春天", "记忆"}

# 核心标签（用于裁剪优先级）
CORE = {"智能体", "多智能体", "世界模型", "强化学习", "推理与规划", "记忆",
        "认知架构", "工具使用", "具身智能", "评估与基准", "游戏智能体",
        "经济模拟", "机器学习", "Web开发", "生活"}

ORDER = ["智能体", "多智能体", "世界模型", "强化学习", "推理与规划", "记忆",
         "认知架构", "工具使用", "具身智能", "评估与基准", "游戏智能体",
         "经济模拟", "机器学习", "Web开发", "生活", "日记",
         "元认知", "自我改进", "技能学习", "长程任务", "上下文管理",
         "信用分配", "知识蒸馏", "空间智能", "物理推理", "视频生成", "策略优化",
         "推荐系统", "矩阵方法", "Web评测", "摄影", "玄武湖", "春天", "记忆"]

MAX_TAGS = 5


def normalize(tags, is_diary=False):
    """映射 -> 去重 -> 裁剪，返回新 tag 列表"""
    mapped = []
    for t in tags:
        t = t.strip()
        if not t:
            continue
        if is_diary and t in DIARY_KEEP:
            nt = t
        else:
            nt = MAP.get(t, t)
        if nt not in mapped:
            mapped.append(nt)
    # 日记类至少「生活」+「日记」两个标签
    if is_diary:
        for must in ("生活", "日记"):
            if must not in mapped:
                mapped.insert(0, must)
    # 至少 2 个标签：不足时从标签名/内容补充领域标签
    if len(mapped) < 2:
        for candidate in ("强化学习", "智能体", "机器学习", "生活"):
            if candidate not in mapped:
                mapped.append(candidate)
                break
    # 校验
    unknown = [t for t in mapped if t not in KEEP]
    if unknown:
        return mapped, unknown
    # 裁剪：核心优先，其次按 ORDER 顺序
    if len(mapped) > MAX_TAGS:
        cores = [t for t in mapped if t in CORE]
        subs = [t for t in mapped if t not in CORE]
        subs.sort(key=lambda x: ORDER.index(x) if x in ORDER else 99)
        kept = (cores + subs)[:MAX_TAGS]
        mapped = kept
    # 按 ORDER 重排
    mapped.sort(key=lambda x: ORDER.index(x) if x in ORDER else 99)
    return mapped, []


def main():
    files = sorted(glob.glob("src/content/posts/*.md"))
    changed = 0
    unknown_all = {}
    for f in files:
        text = open(f, encoding="utf-8").read()
        m = re.match(r"^(---\n)(.*?)(\n---)", text, re.S)
        if not m:
            print(f"[SKIP no-frontmatter] {f}")
            continue
        fm = m.group(2)
        tm = re.search(r"^tags:\s*\[(.*?)\]", fm, re.M)
        if not tm:
            print(f"[SKIP no-tags] {f}")
            continue
        old_raw = tm.group(1)
        old_tags = [t.strip() for t in old_raw.split(",") if t.strip()]
        cm = re.search(r"^category:\s*(.+)$", fm, re.M)
        is_diary = bool(cm and cm.group(1).strip() in ("日记", "杂文"))
        new_tags, unknown = normalize(old_tags, is_diary)
        if unknown:
            unknown_all[f] = unknown
        if new_tags != old_tags:
            changed += 1
            print(f"[{f}]")
            print(f"  - {old_tags}")
            print(f"  + {new_tags}")
            if APPLY:
                new_line = "tags: [" + ", ".join(new_tags) + "]"
                new_fm = fm[:tm.start()] + new_line + fm[tm.end():]
                text = text[:m.start(2)] + new_fm + text[m.end(2):]
                open(f, "w", encoding="utf-8").write(text)
    print(f"\n{'APPLIED' if APPLY else 'DRY-RUN'}: {changed}/{len(files)} files changed")
    if unknown_all:
        print("\n!! UNKNOWN TAGS (need taxonomy update):")
        for f, ts in unknown_all.items():
            print(f"  {f}: {ts}")


if __name__ == "__main__":
    main()
