

import numpy as np

from motmetrics import math_util
# fr = open('/home/btr/ByteTrack/14.mp4.csv', 'rt')
# fw = open('/home/btr/ByteTrack/14.mp4.txt', 'w+')
 
# ls = []
 
# for line in fr:
#     line = line.replace('\n', '')  # 删除每行后面的换行符
#     line = line.split(',')  # 将每行数据以逗号切割成单个字符
#     ls.append(line)  # 将单个字符追加到列表ls中
 
# for row in ls:
#     fw.write(' '.join(row) + '\n')
 
# fr.close()
# fw.close()

def rect_min_max(r):
    min_pt = r[..., :2]
    size = r[..., 2:]
    max_pt = min_pt + size
    return min_pt, max_pt

def boxiou(a, b):
    """Computes IOU of two rectangles."""
    a_min, a_max = rect_min_max(a)
    b_min, b_max = rect_min_max(b)
    # Compute intersection.
    i_min = np.maximum(a_min, b_min)
    i_max = np.minimum(a_max, b_max)
    i_size = np.maximum(i_max - i_min, 0)
    i_vol = np.prod(i_size, axis=-1)
    # Get volume of union.
    a_size = np.maximum(a_max - a_min, 0)
    b_size = np.maximum(b_max - b_min, 0)
    a_vol = np.prod(a_size, axis=-1)
    b_vol = np.prod(b_size, axis=-1)
    u_vol = a_vol + b_vol - i_vol
    return np.where(i_vol == 0, np.zeros_like(i_vol, dtype=np.float64),
                    math_util.quiet_divide(i_vol, u_vol))

a = [331.90,173.73,27.53,40.46]
b = [328,175,33,41]
print(boxiou(np.asfarray(a), np.asfarray(b)))