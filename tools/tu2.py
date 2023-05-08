import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False

# 读取数据
df = pd.read_excel('/home/btr/ByteTrack/ccc.xlsx')

# 获取参数的取值范围
params = df.iloc[:, 0]
plt.rcParams.update({'font.size': 17})

# 循环绘制四个指标的折线图
for i in range(1, 5):
    # 获取当前指标的值
    data = df.iloc[:, i]
    
    # 创建图像对象
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # plt.figure(figsize=(48, 25))
    # 绘制折线图
    # ax = plt.figure()
    ax.plot(params, data)
    
    # 设置图像标题和坐标轴标签
    ax.set_title(df.columns[i])
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[i])
    
    # 显示网格线
    ax.grid()
    
    # 调整y轴间距
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.8 * y_range, y_max + 0.8 * y_range)
    
    # 保存图像
    plt.savefig('/home/btr/ByteTrack/tools/tttu/{}.png'.format(df.columns[i]))
    
    # 显示图像（可选）
    plt.show()