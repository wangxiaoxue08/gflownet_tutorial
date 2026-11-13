"""Offline HTML and standalone SVG charts from measured GFlowNet results.

No extra plotting dependency is required. All probabilities and counts come from
the experiment, and the same SVG assets are embedded in the standalone HTML.
"""

import csv
import json
from html import escape
from pathlib import Path

import numpy as np

COLORS = ("#233e65", "#168478", "#df8a36")


def _text(x: float, y: float, label: str, **attrs) -> str:
    attributes = " ".join(
        f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items()
    )
    return f'<text x="{x:.2f}" y="{y:.2f}" {attributes}>{escape(label)}</text>'


def _svg(content: list[str], title: str, width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{escape(title)}">'
        f'<title>{escape(title)}</title><rect width="100%" height="100%" fill="#fff"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" font-size="12" fill="#334155">'
        + "".join(content)
        + "</g></svg>"
    )


def distribution_chart(report: dict) -> str:
    """Ten categorical probabilities with marginal 95% Wilson sampling intervals."""
    chart = []
    left, top, bottom, width = 68, 58, 350, 920
    maximum = (
        max(max(report[key]) for key in ("target", "model", "interval_high")) * 1.12
    )

    def y(value):
        return bottom - value / maximum * (bottom - top)

    for tick in np.linspace(0, maximum, 6):
        chart.append(
            f'<path d="M{left},{y(tick):.2f} H{left + width}" stroke="#e5eaf0"/>'
        )
        chart.append(_text(left - 9, y(tick) + 4, f"{tick:.1%}", text_anchor="end"))
    for i, (label, color) in enumerate(
        zip(
            ("Target R / Z", "Model (all paths)", "Actual samples + 95% CI"),
            COLORS,
            strict=True,
        )
    ):
        x = left + i * 280
        chart.append(
            f'<rect x="{x}" y="15" width="12" height="12" rx="2" fill="{color}"/>'
        )
        chart.append(_text(x + 19, 26, label))
    for i, subset in enumerate(report["subsets"]):
        center = left + (i + 0.5) * width / 10
        label = "{" + ",".join(map(str, subset)) + "}"
        for j, key in enumerate(("target", "model", "empirical")):
            value = report[key][i]
            x = center + (j - 1) * 21 - 9
            chart.append(
                f'<rect x="{x:.2f}" y="{y(value):.2f}" width="18" '
                f'height="{bottom - y(value):.2f}" fill="{COLORS[j]}">'
                f"<title>{label} · {key}: {value:.8f}</title></rect>"
            )
        x = center + 21
        low, high = y(report["interval_low"][i]), y(report["interval_high"][i])
        chart.append(
            f'<path d="M{x},{low:.2f} V{high:.2f} M{x - 5},{low:.2f} H{x + 5} '
            f'M{x - 5},{high:.2f} H{x + 5}" stroke="#6f4517" fill="none"/>'
        )
        chart.append(_text(center, bottom + 24, label, text_anchor="middle"))
        chart.append(
            _text(
                center,
                bottom + 44,
                f"n={report['counts'][i]}",
                text_anchor="middle",
                fill="#64748b",
            )
        )
    chart.append(
        _text(
            left,
            430,
            f"N = {report['samples']:,} | sampling epsilon = 0 | unordered sets",
        )
    )
    return _svg(chart, "Target, exact model, and sampled set distributions", 1040, 450)


def training_chart(report: dict) -> str:
    """Plot the recorded batch losses and learned log Z; no interpolated measurements."""
    chart = []
    history = report["history"]
    if not history:
        return _svg(
            [_text(40, 60, "No training history in this checkpoint")],
            "Training history",
            1040,
            340,
        )
    steps = np.asarray([row["step"] for row in history], dtype=float)
    for panel, (key, title) in enumerate(
        (
            ("loss", "Batch trajectory-balance loss"),
            ("log_z", "Learned log Z / exact log Z"),
        )
    ):
        values = np.asarray([row[key] for row in history], dtype=float)
        left, top, bottom, width = 65 + panel * 510, 55, 265, 425
        low = (
            min(0.0, float(values.min()))
            if key == "loss"
            else min(float(values.min()), report["log_z_exact"])
        )
        high = (
            float(values.max())
            if key == "loss"
            else max(float(values.max()), report["log_z_exact"])
        )
        span = max(high - low, 1e-8)
        high += span * 0.1
        if key != "loss":
            low -= span * 0.1

        def x(step, left=left, width=width):
            return left + (step - steps[0]) / max(steps[-1] - steps[0], 1) * width

        def y(value, bottom=bottom, low=low, high=high, top=top):
            return bottom - (value - low) / (high - low) * (bottom - top)

        chart.append(_text(left, 26, title, font_weight="600", font_size="15"))
        for tick in np.linspace(low, high, 5):
            chart.append(
                f'<path d="M{left},{y(tick):.2f} H{left + width}" stroke="#e5eaf0"/>'
            )
            chart.append(_text(left - 8, y(tick) + 4, f"{tick:.3g}", text_anchor="end"))
        if key == "log_z":
            chart.append(
                f'<path d="M{left},{y(report["log_z_exact"]):.2f} H{left + width}" stroke="{COLORS[2]}" stroke-dasharray="6 4"/>'
            )
        points = " ".join(
            f"{x(step):.2f},{y(value):.2f}"
            for step, value in zip(steps, values, strict=True)
        )
        chart.append(
            f'<polyline points="{points}" fill="none" stroke="{COLORS[1]}" stroke-width="2"/>'
        )
        for step, value in zip(steps, values, strict=True):
            chart.append(
                f'<circle cx="{x(step):.2f}" cy="{y(value):.2f}" r="2.5" fill="{COLORS[1]}"><title>Step {step:g}: {value:.8g}</title></circle>'
            )
        for tick in np.linspace(steps[0], steps[-1], 5):
            chart.append(
                _text(x(tick), bottom + 21, f"{tick:.0f}", text_anchor="middle")
            )
        chart.append(
            _text(left + width / 2, bottom + 48, "Optimizer step", text_anchor="middle")
        )
    return _svg(chart, "Recorded training loss and log partition function", 1040, 340)


def write_report(report: dict, output: Path) -> Path:
    """Write an offline report, CSV comparison table, and two reusable SVG figures."""
    for key in ("target", "model", "empirical", "interval_low", "interval_high"):
        values = np.asarray(report[key])
        if values.shape != (10,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid report array: {key}")
    if sum(report["counts"]) != report["samples"]:
        raise ValueError("Sample counts do not sum to N")
    distribution, training = distribution_chart(report), training_chart(report)
    (output / "distribution.svg").write_text(distribution, encoding="utf-8")
    (output / "training.svg").write_text(training, encoding="utf-8")
    table_rows = []
    with (output / "distribution.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "set",
                "reward",
                "target",
                "model_exact",
                "count",
                "sample_frequency",
                "ci_low",
                "ci_high",
            ]
        )
        for i, subset in enumerate(report["subsets"]):
            label = "{" + ",".join(map(str, subset)) + "}"
            target, model, empirical = (
                report[key][i] for key in ("target", "model", "empirical")
            )
            low, high = report["interval_low"][i], report["interval_high"][i]
            writer.writerow(
                [
                    label,
                    report["rewards"][i],
                    target,
                    model,
                    report["counts"][i],
                    empirical,
                    low,
                    high,
                ]
            )
            table_rows.append(
                f"<tr><td><code>{label}</code></td><td>{report['rewards'][i]:.6f}</td>"
                f"<td>{target:.4%}</td><td>{model:.4%}</td><td>{report['counts'][i]:,}</td>"
                f"<td>{empirical:.4%}</td><td>{empirical - target:+.4%}</td>"
                f"<td>{low:.3%}–{high:.3%}</td></tr>"
            )
    example_rows = "".join(
        f"<tr><td>{row['index']}</td><td>{' → '.join(map(str, row['actions']))}</td>"
        f"<td><code>{{{','.join(map(str, row['set']))}}}</code></td></tr>"
        for row in report["examples"]
    )
    training_seconds = report["training_seconds_including_compile"]
    duration = (
        f"{training_seconds:.2f} 秒（包含 JIT 编译）"
        if training_seconds is not None
        else "本次直接加载模型，未重新训练"
    )
    restore_error = report["restore_max_probability_error"]
    restore = (
        f"保存前后十个终态概率最大差值 {restore_error:.3g}。"
        if restore_error is not None
        else "从指定 checkpoint 加载参数，在新进程中独立采样。"
    )
    config = report["training_config"]
    sample = report["sampling_config"]
    sample_command = f'uv run python gflownet_nnx_tutorial.py --checkpoint "{output.as_posix()}/model.msgpack" --samples {sample["n"]} --sample-seed {sample["seed"]} --sample-batch-size {sample["batch_size"]}'
    train_command = (
        f"uv run python gflownet_nnx_tutorial.py --steps {config['steps']} "
        f"--batch-size {config['batch_size']} --seed {config['seed']} "
        f"--width {config['width']} --learning-rate {config['learning_rate']} "
        f"--epsilon {config['epsilon']} --backward {config['backward']} "
        f"--log-every {config['log_every']} --samples {sample['n']} "
        f"--sample-seed {sample['seed']} --sample-batch-size {sample['batch_size']}"
    )
    metadata = escape(
        json.dumps(
            {
                "run_id": report["run_id"],
                "created_utc": report["created_utc"],
                "training": config,
                "sampling": sample,
                "versions": report["versions"],
                "hardware": report["hardware"],
                "task": report["task"],
                "source_sha256": report["source_sha256"],
                "checkpoint_sha256": report["checkpoint_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GFlowNet 实验报告 · {escape(report["run_id"])}</title>
<style>
:root{{color-scheme:light;--ink:#20344a;--muted:#5f6e80;--line:#dde5eb;--teal:#168478}}
*{{box-sizing:border-box}}body{{margin:0;background:#eef2f5;color:var(--ink);font:16px/1.8 "Segoe UI","Microsoft YaHei",sans-serif}}
main{{max-width:1160px;margin:32px auto;padding:40px 48px;background:white;border:1px solid var(--line);border-radius:16px}}
h1{{font-size:34px;line-height:1.4;margin:12px 0}}h2{{font-size:23px;margin:0 0 18px}}h3{{font-size:18px}}
section{{border-top:1px solid var(--line);padding-top:30px;margin-top:34px}}p{{margin:12px 0}}.muted,figcaption{{color:var(--muted);font-size:14px}}
.eyebrow{{font-size:12px;letter-spacing:.12em;color:var(--teal);font-weight:700}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:26px 0}}
.card{{border:1px solid var(--line);border-radius:10px;padding:16px;background:#f8fafb}}.card span{{display:block;font-size:13px;color:var(--muted)}}.card strong{{font-size:25px;font-variant-numeric:tabular-nums}}
figure{{margin:18px 0}}svg{{display:block;width:100%;height:auto}}table{{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}}
th,td{{padding:9px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{background:#f1f5f8}}.scroll{{overflow:auto}}
code,pre{{font-family:Consolas,monospace}}pre{{background:#122a3b;color:#e1edf2;padding:18px;border-radius:8px;overflow:auto;font-size:13px;line-height:1.65}}
.note{{padding:14px 18px;background:#eaf4f1;border-left:3px solid var(--teal)}}a{{color:#156b85}}.links{{display:flex;gap:18px;flex-wrap:wrap}}
@media(max-width:760px){{main{{margin:0;padding:24px 18px;border:0;border-radius:0}}h1{{font-size:27px}}.cards{{grid-template-columns:repeat(2,1fr)}}}}
@media print{{body{{background:white}}main{{margin:0;border:0;padding:0}}section{{break-inside:avoid}}}}
</style></head><body><main>
<header><div class="eyebrow">CPU EXPERIMENT / TRAJECTORY BALANCE</div><h1>从 5 个元素中生成 3 元集合</h1>
<p>GFlowNet 训练、模型保存与重新加载、实际采样分布验证。</p>
<p class="muted">运行 {escape(report["run_id"])} · {escape(report["created_utc"])} · {report["samples"]:,} 个样本</p></header>
<div class="cards"><div class="card"><span>模型分布 vs 真实分布 · TV</span><strong>{report["tv_exact"]:.6f}</strong></div>
<div class="card"><span>采样频率 vs 真实分布 · TV</span><strong>{report["tv_empirical"]:.6f}</strong></div>
<div class="card"><span>采样频率 vs 模型分布 · TV</span><strong>{report["tv_sampling"]:.6f}</strong></div>
<div class="card"><span>全 60 条轨迹 · TB RMSE</span><strong>{report["tb_rmse_all_paths"]:.6f}</strong></div></div>
<section><h2>01 / 实验设置与真实分布</h2>
<p>从空集开始，每一步添加一个尚未选择的元素，三步后停止。共有 C(5,3)=10 个终态和 5×4×3=60 条有序轨迹。
奖励偏好 A={{0,1,2}} 和 B={{2,3,4}}，定义距离 d(S,A)=|S △ A|/2。</p>
<pre>R(S) = exp(-2.5 × d(S,A)) + exp(-2.5 × d(S,B))
p*(S) = R(S) / Σ_T R(T)
TB residual = log Z + Σ log P_F - log R(S) - Σ log P_B</pre>
<p>训练使用 {config["backward"]} 反向策略、两层宽度 {config["width"]} 的 SiLU MLP，Adam 学习率 {config["learning_rate"]}，
批量 {config["batch_size"]}，{config["steps"]:,} 次更新，探索比例 ε={config["epsilon"]}。仅使用在线采到的轨迹及其终态奖励；
所有终态和轨迹的枚举只用于训练完成后的验证，没有外部数据集、验证集选模或测试集调参。</p>
<p>训练耗时：{duration}。这是单次 CPU 实验记录，不是性能基准。</p></section>
<section><h2>02 / 分布对比</h2><figure>{distribution}<figcaption>蓝色为枚举奖励得到的真实分布；绿色为对所有前向路径概率求和得到的模型精确分布；橙色为实际样本频率。
误差条是逐类别的 95% Wilson 区间，用于估计模型采样概率，不是模型相对真实分布的置信区间，也不是同时覆盖十个类别的置信带。</figcaption></figure>
<p>总变差距离 TV(p,q)=½Σ|p−q|，范围为 0–1。模型 TV 衡量训练误差；采样对模型的 TV 反映有限次随机抽样的波动。</p>
<div class="scroll"><table><thead><tr><th>集合</th><th>奖励 R</th><th>真实概率</th><th>模型概率</th><th>样本数</th><th>样本频率</th><th>频率−真实</th><th>95% 区间</th></tr></thead><tbody>{"".join(table_rows)}</tbody></table></div></section>
<section><h2>03 / 实际生成的集合样例</h2><p>下面是本次采样的前 {len(report["examples"])} 条原始结果，保留重复集合，不按结果挑选。
不同生成顺序可以对应同一个无序集合。统计分布使用全部 {report["samples"]:,} 个样本。</p>
<div class="scroll"><table><thead><tr><th>样本编号</th><th>添加顺序</th><th>最终集合</th></tr></thead><tbody>{example_rows}</tbody></table></div></section>
<section><h2>04 / 训练记录与保存验证</h2><figure>{training}<figcaption>点为实际记录的训练批次损失及更新后的 log Z，连线只用于阅读；损失是更新前的批次均值，不是全轨迹误差。
右图虚线为精确 log Z。</figcaption></figure>
<p>学得 log Z={report["log_z_learned"]:.8f}；精确 log Z={report["log_z_exact"]:.8f}。
全部前向路径的概率和为 {report["forward_path_sum"]:.9f}；全轨迹最大 |TB residual|={report["tb_max_abs_all_paths"]:.6f}。</p>
<p class="note">{restore}本报告中的样本来自重新加载后的模型，采样探索比例 ε=0。
model.msgpack 保存全部模型变量、架构、训练配置和历史，供加载推断；不包含优化器和训练 RNG 状态，不支持精确断点续训。</p>
<p>单次固定种子训练不代表多种子稳定性；结论仅适用于这个可枚举的小规模任务。</p></section>
<section><h2>05 / 复现与文件</h2><p>在项目目录使用锁定的 uv 环境：</p><pre>uv sync --locked
{escape(train_command)}

# 独立进程加载本次模型并采样（自动创建新输出目录）
{escape(sample_command)}</pre>
<div class="links"><a href="model.msgpack">训练后的模型</a><a href="samples.csv">全部生成样本 CSV</a><a href="distribution.csv">分布对比 CSV</a><a href="metrics.json">原始指标 JSON</a><a href="distribution.svg">分布图 SVG</a><a href="training.svg">训练图 SVG</a></div>
<p>GitHub 源码：<a href="https://github.com/wangxiaoxue08/gflownet_tutorial/blob/master/gflownet_nnx_tutorial.py"><code>gflownet_nnx_tutorial.py</code></a> · <a href="https://github.com/wangxiaoxue08/gflownet_tutorial/blob/master/utility.py"><code>utility.py</code></a> · <a href="https://github.com/wangxiaoxue08/gflownet_tutorial/blob/master/experiment_report.py"><code>experiment_report.py</code></a>。</p>
<p class="muted">HTML 内嵌图表，可以单独离线阅读；下载链接需与同目录文件一起保留。运行目录还保存脚本、pyproject.toml 和 uv.lock 快照。</p>
<details><summary>配置、版本、硬件与文件校验</summary><pre>{metadata}</pre></details></section>
</main></body></html>"""
    path = output / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
