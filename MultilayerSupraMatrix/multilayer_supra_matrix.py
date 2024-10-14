import pandas as pd
import numpy as np
import math
from .config import CONFIGFILE, CONFIGPARAMS

# Define a class to represent a space
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
    
        for city in CONFIGPARAMS.cities:
            if city in self.data.keys() and city not in self.data[city].keys():
                self.add_edge(city, city,0)

def interlayer_edge_weight(start_layer, end_layer, dic_distance, para_list):
    lambda_virtual, lambda_physical, t_virtual, t_physical = para_list
    weight_dic = {'virtual2physical':{}, 'physical2virtual':{}}
    
    if start_layer.type == 'virtual':
        for start_city in CONFIGPARAMS.cities:
            weight_dic['virtual2physical'][start_city] = {}
            for end_city in CONFIGPARAMS.cities:
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
                        a = non_corresponding_interactions(start_city, end_city, start_layer, end_layer, dic_distance, para_list)
                weight_dic['virtual2physical'][start_city][end_city] = a
        return weight_dic['virtual2physical']
    
    elif start_layer.type == 'physical':
        for start_city in CONFIGPARAMS.cities:
            weight_dic['physical2virtual'][start_city] = {}
            for end_city in CONFIGPARAMS.cities:
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
                        a = non_corresponding_interactions(start_city, end_city, start_layer, end_layer, dic_distance, para_list)
                weight_dic['physical2virtual'][start_city][end_city] = a
        return weight_dic['physical2virtual']

def corresponding_interactions(start_city, end_city, start_layer, end_layer):
    vector1 = np.array([start_layer.data[start_city][i+1] if i+1 in start_layer.data[start_city].keys() else 0 for i in range(len(CONFIGPARAMS.cities))])
    vector2 = np.array([end_layer.data[end_city][i+1] if i+1 in end_layer.data[end_city].keys() else 0 for i in range(len(CONFIGPARAMS.cities))])

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

def non_corresponding_interactions(start_city, end_city, start_layer, end_layer, dic_distance, para_list):
    lambda_virtual, lambda_physical, t_virtual, t_physical = para_list
    if start_layer.type == 'virtual':
        weight_distance = math.exp(-(dic_distance[start_city][end_city])/lambda_virtual)
        p1 = start_layer.data[start_city][end_city] / sum(list(start_layer.data[start_city].values()))
        p2 = end_layer.data[start_city][end_city] / sum(list(end_layer.data[start_city].values()))
        c = 1 - abs(p1 - p2)
        # print(-(dic_distance[start_city][end_city])/lambda_physical)
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

        # print(-(dic_distance[start_city][end_city])/lambda_physical)
        return weight_distance * c

def dic_to_matrix(dic, diagonal=False):
    matrix = np.zeros((len(CONFIGPARAMS.cities), len(CONFIGPARAMS.cities)))
    for i in range(len(CONFIGPARAMS.cities)):
        if diagonal == True:
            for j in range(len(CONFIGPARAMS.cities)):
                matrix[i][j] = dic[CONFIGPARAMS.cities[i]][CONFIGPARAMS.cities[j]]
        if diagonal == False:
            for j in range(len(CONFIGPARAMS.cities)):
                if i == j:
                    matrix[i][j] = 0
                elif CONFIGPARAMS.cities[i] not in dic.keys() or CONFIGPARAMS.cities[j] not in dic[CONFIGPARAMS.cities[i]].keys():
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = dic[CONFIGPARAMS.cities[i]][CONFIGPARAMS.cities[j]]
    return matrix

def caculate_supra_matrix(time):
    # Initialization parameters
    configfile = CONFIGFILE(time)
    dic_distance = CONFIGPARAMS.cityDistance()
    print(f"Start caculate supramatrix for {time}")

    # Create a virtual layer instance
    virtual = Space('virtual')
    physical = Space('physical')

    # Read edge information
    virtual.load_from_file(configfile.virtual_file_path)
    physical.load_from_file(configfile.physical_file_path)

    values_virtual = [value for sub_dict in virtual.data.values() for value in sub_dict.values()]
    t_virtual = sum(values_virtual) / (CONFIGPARAMS.city_num * CONFIGPARAMS.city_num)
    values_physical = [value for sub_dict in physical.data.values() for value in sub_dict.values()]
    t_physical = sum(values_physical) / (CONFIGPARAMS.city_num * CONFIGPARAMS.city_num)

    lambda_dict = CONFIGPARAMS.getlambda_dict()
    lambda_virtual = lambda_dict["virtual"][time]
    lambda_physical = lambda_dict["physical"][time]

    print(f"Finish read edge information")

    # paralist: store lambda_virtual, lambda_physical, t_virtual, t_physical
    para_list = [lambda_virtual, lambda_physical, t_virtual, t_physical]

    # Calculate the superamatrix and store
    dic_virtual = virtual.data
    dic_physical = physical.data
    dic_virtual2physical = interlayer_edge_weight(virtual, physical, dic_distance, para_list)
    dic_physical2virtual = interlayer_edge_weight(physical, virtual, dic_distance, para_list)

    with open(configfile.virtual_interlayer_file_path, "w") as f:
        for key in dic_virtual2physical.keys():
            for sub_key in dic_virtual2physical[key].keys():
                f.write(f"{key} {sub_key} {dic_virtual2physical[key][sub_key]}\n")
    with open(configfile.physical_interlayer_file_path, "w") as f:
        for key in dic_physical2virtual.keys():
            for sub_key in dic_physical2virtual[key].keys():
                f.write(f"{key} {sub_key} {dic_physical2virtual[key][sub_key]}\n")

    matrix1 = dic_to_matrix(dic_virtual)
    matrix2 = dic_to_matrix(dic_virtual2physical, True)
    matrix3 = dic_to_matrix(dic_physical2virtual, True)
    matrix4 = dic_to_matrix(dic_physical)

    result_supra_matrix = np.vstack((np.hstack((matrix1, matrix2)), np.hstack((matrix3, matrix4))))
    df = pd.DataFrame(result_supra_matrix)
    df.to_csv(configfile.supramatrix_file_path)

    print(f"Finish caculate supramatrix for {time}")