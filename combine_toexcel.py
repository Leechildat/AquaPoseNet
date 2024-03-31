import json
import numpy as np
import pandas as pd


# 点到直线距离
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


joints_eng = ['M', 'H1', 'H2', 'A1', 'A2', 'W1', 'W2', 'W3', 'Tail', 'Eh', 'Et']
with open('json/fish_blank_2400.json', mode='r') as json_file:
    data = json.load(json_file)['images']

with open('result/keypoints_fish_eyes_blank_7.14_results.json', 'r') as f:
    eye_json = json.load(f)

with open('result/keypoints_fish_blank_2400_results.json', 'r') as f:
    fish_json = json.load(f)

# id 检索图片名和索引
dic_data = {}
for item in data:
    dic_idx = {}
    dic_idx['file_name'] = item['file_name']
    dic_data[item['id']] = dic_idx
for idx, item in enumerate(eye_json):
    image_id = item['image_id']
    dic_data[image_id]['eye_idx'] = idx
for idx, item in enumerate(fish_json):
    image_id = item['image_id']
    dic_data[image_id]['fish_idx'] = idx

# TODO: 组合两个json文件，生成excel

column = [f'X{i}' for i in range(1, 10)]
index = []  # 文件名
data = []

for i in dic_data.keys():
    data_image = [0 for i in range(9)]
    index.append(dic_data[i]['file_name'])
    eye_idx = dic_data[i]['eye_idx']
    fish_idx = dic_data[i]['fish_idx']
    fish_points = np.array(fish_json[fish_idx]['keypoints']).reshape(9, 3)[:, :2].tolist()
    eye_points = np.array(eye_json[eye_idx]['keypoints']).reshape(2, 3)[:, :2].tolist()

    data_image[2] = point_distance_line(
        np.array(fish_points[0]),
        np.array(fish_points[1]),
        np.array(fish_points[2])
    )
    bd = point_distance_line(
        np.array(eye_points[0]),
        np.array(fish_points[1]),
        np.array(fish_points[2])
    )
    data_image[3] = data_image[2] - bd
    data_image[4] = point_distance(eye_points[0], eye_points[1])
    # X6不能计算
    data_image[5] = 0
    data_image[6] = point_distance(fish_points[3], fish_points[4])
    data_image[7] = point_distance_line(
        np.array(fish_points[7]),
        np.array(fish_points[5]),
        np.array(fish_points[6])
    )
    data_image[8] = point_distance(fish_points[5], fish_points[6])
    data_image[1] = data_image[2] + point_distance_line(
        np.array(fish_points[1]),
        np.array(fish_points[3]),
        np.array(fish_points[4])
    ) + point_distance_line(
        np.array(fish_points[7]),
        np.array(fish_points[3]),
        np.array(fish_points[4])
    )
    data_image[0] = data_image[1] + point_distance(fish_points[7], fish_points[8])
    data.append(data_image)

df = pd.DataFrame(data=data, index=index, columns=column)

df.to_excel(excel_writer='demo.xlsx', sheet_name='sheet_1')

print("done")

