import argparse
import os
import os.path as osp
import time
from turtle import down
import cv2
import numpy as np
import torch

CLASSES = ['baishi_500',
'baishi_330',
'baishi_black_330',
'baisuishan',
'dongfangshuye',
'dongpeng',
'hongniu',
'ksf_binghongcha',
'ksf_lvcha',
'ksf_molimi',
'ksf_moliqing',
'ksf_qinmeilvcha',
'kekoukele_330',
'kekoukele_600',
'maidong',
'nongfushanquan',
'shuirongc100',
'shanzhashuxia',
'asamu_nailv',
'asamu_naicha',
'guolicheng',
'xuebi_330',
'yezizhi',
'yuanqi',
'yuanqi_waixingren'
]

def get_polygon(width, height):
        # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    # 初始化2个撞线polygon
    # list_pts_blue = [[50, 525],  [1898, 650], [1893, 750], [45, 625]]
    # list_pts_blue = [[45, 475],  [1898, 600], [1893, 1000], [45, 875]]
    list_pts_blue = [[45, 600],  [1898, 600], [1893, 900], [45, 900]]
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]
    # 填充第二个polygon
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    # list_pts_yellow = [[50, 425],  [1898, 550], [1893, 650], [45, 525]]
    # list_pts_yellow = [[45, 75],  [1898, 200], [1893, 600], [45, 475]]
    list_pts_yellow = [[45, 300],  [1898, 300], [1893, 600], [45, 600]]
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]
    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask = polygon_blue_value_1 + polygon_yellow_value_2
    # 缩小尺寸
    polygon_mask = cv2.resize(polygon_mask, (int(width), int(height)))
    # 蓝 色盘 b,g,r
    blue_color_plate = [0, 255, 255]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)
    # 黄 色盘
    yellow_color_plate = [255, 0, 0]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)
    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image
    # 缩小尺寸，1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, (int(width), int(height)))

    return polygon_mask, color_polygons_image

def judge(online_targets, list_overlapping_blue_polygon, list_overlapping_yellow_polygon, down_count, up_count, class_name, polygon_mask_blue_and_yellow, cart):
    list_bboxs = []
    for tracker in online_targets:
        x1, y1, x2, y2 = tracker.tlbr
        track_id = tracker.track_id
        list_bboxs.append((x1, y1, x2, y2, class_name, track_id))
    for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id = item_bbox

                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * 0.6))

                # 撞线的点
                y = y1_offset
                x = int(x1)

                if polygon_mask_blue_and_yellow[y, x] == 1:
                    # 如果撞 蓝polygon
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)
                    pass

                    # 判断 黄polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 外出方向
                    if track_id in list_overlapping_yellow_polygon:
                        # 外出+1
                        up_count += 1

                        print(f'类别: {CLASSES[label]} | id: {track_id} | 入柜撞线 | 入柜撞线总数: {up_count} | 入柜id列表: {list_overlapping_yellow_polygon}')

                        # 删除 黄polygon list 中的此id
                        list_overlapping_yellow_polygon.remove(track_id)
                        if cart[label] < 1:
                            print("异常！")
                        else:
                            cart[label] -= 1

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass

                elif polygon_mask_blue_and_yellow[y, x] == 2:
                    # 如果撞 黄polygon
                    if track_id not in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.append(track_id)
                    pass

                    # 判断 蓝polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 进入方向
                    if track_id in list_overlapping_blue_polygon:
                        # 进入+1
                        down_count += 1

                        print(f'类别: {CLASSES[label]} | id: {track_id} | 离柜撞线 | 离柜撞线总数: {down_count} | 离柜id列表: {list_overlapping_blue_polygon}')

                        # 删除 蓝polygon list 中的此id
                        list_overlapping_blue_polygon.remove(track_id)
                        
                        cart[label] += 1

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass
                    pass
                else:
                    pass
                pass

    return list_overlapping_blue_polygon, list_overlapping_yellow_polygon, down_count, up_count, cart

def plot_text(online_im, cart):
    im = np.ascontiguousarray(np.copy(online_im))
    p = 3
    for i in range(0, len(cart)):
        if cart[i] is not 0:
            cv2.putText(im, '%s : %d' % (CLASSES[i], cart[i]),
                (0, int(15 * p)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
            p += 1
    return im