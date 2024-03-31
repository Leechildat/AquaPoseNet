import json
import os.path
import math
import csv

import numpy as np
import pandas as pd
from pycocotools import mask


def ad(point1, point2, size):
    vec = point2 - point1
    vec_perp1 = np.array([-vec[1], vec[0]])
    vec_perp2 = np.array([vec[1], -vec[0]])
    if vec_perp1[1] == 0:
        a = (point1[0], 0)
        b = (point1[0], size[1])
    p1 = vec_perp1 + point1
    p2 = vec_perp2 + point1
    abs_x = point2[0] - point1[0]
    size


def get_foot(point, point1, point2):
    start_x, start_y = point1[0], point1[1]
    end_x, end_y = point2[0], point2[0]
    pa_x, pa_y = point

    p_foot = [0, 0]
    if start_x == end_y:
        p_foot[0] = start_x
        p_foot[1] = point[1]
        return p_foot
    k = (end_y - start_y) * 1.0 / (end_x - start_x)
    a = k
    b = -1.0
    c = start_y - k * start_x
    p_foot[0] = (b * b * pa_x - a * b * pa_y - a * c) / (a * a + b * b)
    p_foot[1] = (a * a * pa_y - a * b * pa_x - b * c) / (a * a + b * b)
    return p_foot


def point_distance_line(point, line_point1, line_point2):
    # 计算向量
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    return distance


def point_distance(point1, point2):
    x1 = point1[0]
    x2 = point2[0]
    y1 = point1[1]
    y2 = point2[1]
    return pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5)


joints_eng = ['M', 'Eh', 'Et', 'G', 'H', 'A1', 'A2', 'W1', 'W2', 'W3', 'T']

with open('output/ModelFishPose/LiteHRNet_50/model_256x256_VMHRNet18_22F2_small'
          '/results/keypoints_test_results.json', 'r') as f:
    fish_json = json.load(f)

with open(r'/mnt/d/data/Fisheries Institute/rules/rule.json', 'r') as f:
    rule_json = json.load(f)

with open(r'/mnt/d/data/Fisheries Institute/result_json/fish_adult_segm_result.json', 'r') as f:
    segm_json = json.load(f)

with open(r'/mnt/d/data/Fisheries Institute/test_json/fish_adult_segm_test.json', 'r') as f:
    name_json = json.load(f)["images"]

df = pd.read_csv(r'/mnt/d/PycharmProjects/fish_weight/csvData/manuals/mask_2D_feature_3375dp_16dv_0917.csv')
column1_data = df.iloc[:, 0]
column2_data = df.iloc[:, 1]


print(len(fish_json))
print(len(rule_json))
print(len(segm_json))
print(len(name_json))
dic_data = {}

bin_mask = mask.decode(segm_json[0]['segmentation'])
bin_mask[bin_mask > 0] = 255
print(bin_mask.shape)

column = [f'X{i}' for i in range(1, 10)]
index = []  # 文件名
data = []

for item in fish_json:
    name = item['image_name']
    for i in range(len(rule_json)):
        rule_name = os.path.basename(rule_json[i]['name'])
        if rule_name == name:
            rule = rule_json[i]['rule']
            break
    # for k_name in name_json:
    #     segm_name = k_name["file_name"].split("\\")[-1]
    #     if segm_name == name:
    #         for j_segm in segm_json:
    #             if k_name["id"] == j_segm["image_id"]:
    #                 bin_mask = mask.decode(j_segm['segmentation'])
    #                 bin_mask[bin_mask > 0] = 255
    #             else:
    #                 continue


    data_image = [0 for i in range(11)]
    for j in range(len(column1_data)):
        if column1_data[j] == name:
            data_image[9] = name
            data_image[10] = column2_data[j]
    fish_points = np.array(item['keypoints']).reshape(11, 3)[:, :2]
    data_image[2] = point_distance(fish_points[0], fish_points[4])
    bd = point_distance(fish_points[1], fish_points[4])
    data_image[3] = data_image[2] - bd
    data_image[4] = point_distance(fish_points[1], fish_points[2])

    h_foot = get_foot(fish_points[3], fish_points[5], fish_points[6])
    # data_image[5] = find_intersection_length_with_direction(fish_points[5], fish_points[6], fish_points[3], bin_mask)
    data_image[6] = point_distance(fish_points[5], fish_points[6])
    data_image[7] = point_distance_line(fish_points[9], fish_points[7], fish_points[8])
    data_image[8] = point_distance(fish_points[7], fish_points[8])
    data_image[1] = (data_image[2] + point_distance_line(fish_points[4], fish_points[5], fish_points[6])
                     + point_distance_line(fish_points[9], fish_points[5], fish_points[6]))
    data_image[0] = data_image[1] + point_distance(fish_points[9], fish_points[10])

    for i in range(len(data_image) - 2):
        data_image[i] = data_image[i] / rule


    # print(data_image)
    data.append(data_image)

filename = 'example.csv'

# 将数据写入CSV文件
with open(filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(data)

print("finished writing")
