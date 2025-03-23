import matplotlib.pyplot as plt
import numpy as np

# ================= 配置参数 =================
# 自定义数据（支持任意多组数据）
datasets = [
    {
        'x': [3 * i + 3 for i in range(20)],  # 横坐标数据
        'y': [41147.08332639386, 43422.47307812696, 44766.33158181753, 45936.174841969536, 47012.08450240636, 48242.85352190036, 49205.18761258134, 50005.65130616464, 50835.05375633856, 51539.38068748334, 51896.652284895266, 51384.92281191928, 51602.090223230836, 51594.22896329567, 51406.357686357565, 51816.90196710247, 51712.67246405657, 51583.16792651861, 51638.42202770639, 51320.47632532731],
        'label': 'dijkstra',  # 数据标签（必需）
        'color': '#2ca02c',  # 线条颜色（可选）
        'marker': 'D'  # 数据点标记（可选）
    },
    {
        'x': [3 * i + 3 for i in range(20)],
        'y': [41026.08331624386, 41498.47307812696, 42879.33158181753, 44487.174841969536, 46454.08450240636, 48078.85352190036, 48970.18761258134, 50294.65130616464, 50716.05375633856, 50945.38068748334, 51107.652284895266, 50301.92281191928, 50978.090223230836, 50754.22896329567, 51149.357686357565, 50898.90196710247, 51198.67246405657, 50946.16792651861, 51007.42202770639, 51034.47632532731],
        'label': 'NSGA2 for shortest distance',
        'linestyle': '--'
    },
    # {
    #     'x': [3 * i + 3 for i in range(20)],
    #     'y': [0, 3257.5790000003217, 6778.89339478220, 10073.2206952855, 13034.6171366827, 16019.8718691232, 18754.5411370561, 21355.3134017826, 23492.2132715105, 25353.9858441122, 27142.1679021998, 28823.9674527737, 30497.8378637174, 32083.3421843070, 33433.2136236985, 34730.8678791964, 35962.0050926505, 37367.5780507212, 38671.2040104033, 40185.5992560081],
    #     'label': 'NSGA2 for lowest generation cost',
    #     'marker': '^'
    # }
]

# 图表元数据配置
chart_config = {
    'title': 'total electric cost of different algorithms',
    'xlabel': 'time / min',
    'ylabel': 'traffic cost',
    'figsize': (10, 6),
    'grid': True,
    'legend_loc': 'upper left'
}
# ===========================================

# 创建画布
plt.figure(figsize=chart_config['figsize'])

# 颜色自动生成（当未指定颜色时）
colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

# 遍历绘制所有数据集
for i, dataset in enumerate(datasets):
    # 合并默认样式与自定义样式
    style = {
        'marker': 'o',
        'linestyle': '-',
        'linewidth': 2,
        'markersize': 8,
        'color': colors[i]
    }
    style.update({k: v for k, v in dataset.items() if k in style})

    # 绘制单条折线
    plt.plot(
        dataset['x'],
        dataset['y'],
        label=dataset['label'],
        **style
    )

# 自动计算坐标轴范围
all_x = [item for d in datasets for item in d['x']]
all_y = [item for d in datasets for item in d['y']]
plt.xlim(min(all_x) - 0.5, max(all_x) + 0.5)
plt.ylim(min(all_y) * 0.9, max(all_y) * 1.1)

# 添加图表元素
plt.title(chart_config['title'], fontsize=14, pad=20)
plt.xlabel(chart_config['xlabel'], fontsize=12)
plt.ylabel(chart_config['ylabel'], fontsize=12)

if chart_config['grid']:
    plt.grid(True, linestyle='--', alpha=0.6)

plt.legend(
    loc=chart_config['legend_loc'],
    fontsize=10,
    framealpha=0.9
)

# 优化布局
plt.tight_layout()
plt.show()

# 保存图表（可选）
# plt.savefig('multi_line_chart.png', dpi=300, bbox_inches='tight')