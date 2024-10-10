import pandas as pd
import numpy as np
import math

global cities, t_virtual, t_physical, lambda_virtual, lambda_physical
global dic_distance

# 定义表示空间的类
class Space:
    def __init__(self, type):
        self.data = {}
        self.type = type
    
    def add_edge(self, start_city, end_city, weight):
        if start_city not in self.data:
            self.data[start_city] = {}
        self.data[start_city][end_city] = weight
    
    def load_from_file(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                start_city, end_city, weight = line.strip().split()
                self.add_edge(int(start_city), int(end_city), float(weight))
    
        for city in cities:
            if city in self.data.keys() and city not in self.data[city].keys():
                self.add_edge(city, city,0)

def distance_calculate():
    df = pd.read_csv("/Users/wishingtree/Desktop/learn_python/MNA/interlayer_edge/city_xy.csv", header=None, index_col=None)
    dic_xy = {row[0]: (row[1], row[2]) for row in df.itertuples(index=False)}
    for start_city in dic_xy.keys():
        dic_distance[start_city] = {}
        for end_city in dic_xy.keys():
            x1, y1 = dic_xy[start_city]
            x2, y2 = dic_xy[end_city]
            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            dic_distance[start_city][end_city] = distance
    return dic_distance

def interlayer_edge_weight(start_layer, end_layer):
    weight_dic = {'virtual2physical':{}, 'physical2virtual':{}}
    
    if start_layer.type == 'virtual':
        for start_city in cities:
            weight_dic['virtual2physical'][start_city] = {}
            for end_city in cities:
                if start_city not in start_layer.data.keys() or (start_city != end_city and end_city not in start_layer.data[start_city].keys()):
                    a = 0
                elif start_city not in end_layer.data.keys() or (start_city != end_city and end_city not in end_layer.data[start_city].keys()):
                    a = 0
                # 虚到实的对应节点互动
                elif start_city == end_city:
                    a = corresponding_interactions(start_city, end_city, start_layer, end_layer)
                # 虚到实的非对应节点互动
                else:
                    if start_layer.data[start_city][end_city] <= t_virtual:
                        a = 0
                    else:
                        a = non_corresponding_interactions(start_city, end_city, start_layer, end_layer)
                weight_dic['virtual2physical'][start_city][end_city] = a
        return weight_dic['virtual2physical']
    
    elif start_layer.type == 'physical':
        for start_city in cities:
            weight_dic['physical2virtual'][start_city] = {}
            for end_city in cities:
                if start_city not in start_layer.data.keys() or (start_city != end_city and end_city not in start_layer.data[start_city].keys()):
                    a = 0
                elif start_city not in end_layer.data.keys() or (start_city != end_city and end_city not in end_layer.data[start_city].keys()):
                    a = 0
                # 实到虚的对应节点互动
                elif start_city == end_city:
                    a = corresponding_interactions(start_city, end_city, start_layer, end_layer)
                # 实到虚的非对应节点互动
                else:
                    if start_layer.data[start_city][end_city] <= t_physical:
                        a = 0
                    else:
                        a = non_corresponding_interactions(start_city, end_city, start_layer, end_layer)
                weight_dic['physical2virtual'][start_city][end_city] = a
        return weight_dic['physical2virtual']

def corresponding_interactions(start_city, end_city, start_layer, end_layer):
    vector1 = np.array([start_layer.data[start_city][i+1] if i+1 in start_layer.data[start_city].keys() else 0 for i in range(len(cities))])
    vector2 = np.array([end_layer.data[end_city][i+1] if i+1 in end_layer.data[end_city].keys() else 0 for i in range(len(cities))])

    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        c = 0
    else:
        cosine_similarity = dot_product / (norm1 * norm2)
        # c = (1 + cosine_similarity) / 2
        c = cosine_similarity
    return c

def non_corresponding_interactions(start_city, end_city, start_layer, end_layer):
    if start_layer.type == 'virtual':
        weight_distance = math.exp(-(dic_distance[start_city][end_city])/lambda_virtual)
        p1 = start_layer.data[start_city][end_city] / sum(list(start_layer.data[start_city].values()))
        p2 = end_layer.data[start_city][end_city] / sum(list(end_layer.data[start_city].values()))
        c = 1 - abs(p1 - p2)
        print(-(dic_distance[start_city][end_city])/lambda_physical)
        return weight_distance * c
    
    else:
        weight_distance = math.exp(-(dic_distance[start_city][end_city])/lambda_physical)
        q1 = list(start_layer.data.keys())
        # q1.remove(end_city)
        q2 = list(end_layer.data.keys())
        # q2.remove(end_city)

        denominator1 = sum([start_layer.data[i][end_city] for i in q1 if end_city in start_layer.data[i].keys()])
        denominator2 = sum([end_layer.data[i][end_city] for i in q2 if end_city in end_layer.data[i].keys()])

        p1 = 0 if denominator1 == 0 else start_layer.data[start_city][end_city] / denominator1
        p2 = 0 if denominator2 == 0 else end_layer.data[start_city][end_city] / denominator2

        c = 1 - abs(p1 - p2)

        print(-(dic_distance[start_city][end_city])/lambda_physical)
        return weight_distance * c

def dic_to_matrix(dic, diagonal=False):
    matrix = np.zeros((len(cities), len(cities)))
    for i in range(len(cities)):
        if diagonal == True:
            for j in range(len(cities)):
                matrix[i][j] = dic[cities[i]][cities[j]]
        if diagonal == False:
            for j in range(len(cities)):
                if i == j:
                    matrix[i][j] = 0
                elif cities[i] not in dic.keys() or cities[j] not in dic[cities[i]].keys():
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = dic[cities[i]][cities[j]]
    return matrix
    
if __name__ == '__main__':
    cities = [i+1 for i in range(371)]

    virtual_file_path = "/Users/wishingtree/Desktop/learn_python/MNA/interlayer_edge/data_0830/search_2023_normalized.edges"
    physical_file_path = "/Users/wishingtree/Desktop/learn_python/MNA/interlayer_edge/data_0830/travel_2023_normalized.edges"
    virtual_interlayer_file_path = "/Users/wishingtree/Desktop/learn_python/MNA/interlayer_edge/res_normalized/edge/interlayer_v_2023.edges"
    physical_interlayer_file_path = "/Users/wishingtree/Desktop/learn_python/MNA/interlayer_edge/res_normalized/edge/interlayer_p_2023.edges"
    result_file_path = "/Users/wishingtree/Desktop/learn_python/MNA/interlayer_edge/res_normalized/matrix/original/result_supra_matrix_2023.csv"

    # 计算城市间地理距离
    dic_distance = {}
    dic_distance = distance_calculate()
    keys = list(dic_distance.keys())
    dis_matrix = np.zeros((371, 371))
    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            dis_matrix[i][j] = dic_distance[key1][key2]
    df_dis = pd.DataFrame(dis_matrix)
    df_dis.to_csv("distance_matrix.csv", index=False, header=False)

    # 创建虚实实例
    virtual = Space('virtual')
    physical = Space('physical')

    # 读取边文件信息
    virtual.load_from_file(virtual_file_path)
    physical.load_from_file(physical_file_path)

    values_virtual = [value for sub_dict in virtual.data.values() for value in sub_dict.values()]
    p1 = np.percentile(values_virtual, 5)
    p2 = np.median(values_virtual)
    p3 = np.percentile(values_virtual, 88)
    p4 = sum(values_virtual) / (371 * 371)
    print(p1, p2, p3, p4)
    t_virtual = p4

    values_physical = [value for sub_dict in physical.data.values() for value in sub_dict.values()]
    q1 = np.percentile(values_physical, 5)
    q2 = np.median(values_physical)
    q3 = np.percentile(values_physical, 95)
    q4 = sum(values_physical) / (371 * 371)
    print(q1, q2, q3, q4)
    t_physical = q4

    lambda_virtual = 200000
    lambda_physical = 100000

    dic_virtual = virtual.data
    dic_physical = physical.data
    dic_virtual2physical = interlayer_edge_weight(virtual, physical)
    dic_physical2virtual = interlayer_edge_weight(physical, virtual)
    print(len(dic_virtual2physical), len(dic_physical2virtual))

    with open(virtual_interlayer_file_path, "w") as f:
        for key in dic_virtual2physical.keys():
            for sub_key in dic_virtual2physical[key].keys():
                f.write(f"{key} {sub_key} {dic_virtual2physical[key][sub_key]}\n")
    with open(physical_interlayer_file_path, "w") as f:
        for key in dic_physical2virtual.keys():
            for sub_key in dic_physical2virtual[key].keys():
                f.write(f"{key} {sub_key} {dic_physical2virtual[key][sub_key]}\n")

    matrix1 = dic_to_matrix(dic_virtual)
    matrix2 = dic_to_matrix(dic_virtual2physical, True)
    matrix3 = dic_to_matrix(dic_physical2virtual, True)
    matrix4 = dic_to_matrix(dic_physical)

    print(p1, p2, p3, p4)
    print(q1, q2, q3, q4)
    
    result_supra_matrix = np.vstack((np.hstack((matrix1, matrix2)), np.hstack((matrix3, matrix4))))
    df = pd.DataFrame(result_supra_matrix)
    df.to_csv(result_file_path)