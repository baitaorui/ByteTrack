import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import random

# 设置默认字体和字号
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 25
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 读取Excel数据
df = pd.read_excel('vvvv.xlsx')

# 生成颜色列表
colors = ['#'+ ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(df))]

# 设置图形大小
plt.figure(figsize=(47, 22))

# 绘制柱状图
bars = plt.bar(range(len(df)), df['数量'], color=colors)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height), 
                 xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

# 隐藏横坐标
plt.xticks([])

# 添加图例
plt.legend(bars, df['类别'], loc='center left', bbox_to_anchor=(1, 0.5))

# 添加标题和标签
plt.title('Number of Classes',fontsize=35)
plt.xlabel('class',fontsize=35)
plt.ylabel('number',fontsize=35)

# 显示图形
plt.show()
# 显示图形
plt.show()
# 保存柱状图
plt.savefig('柱状图.png')



