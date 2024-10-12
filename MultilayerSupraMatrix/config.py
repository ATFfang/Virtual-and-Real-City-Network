import pandas as pd
import math
import csv
global ROOT_DIR 
ROOT_DIR = "E:\Project\Academic\虚实网络\Virtual and Real City Network\DATA\\"

class CONFIGFILE():
    def __init__(self, time: int):
        self.time = time

        # root directory
        ROOT_DIR = "E:\Project\Academic\虚实网络\Virtual and Real City Network\DATA\\"

        # city xy file path
        self.cityXY_filepath = ROOT_DIR + "rawData\\city_xy.csv"

        # virtual/physical network file path
        self.virtual_file_path = ROOT_DIR + f"rawData\\search_{self.time}_normalized.edges"
        self.physical_file_path = ROOT_DIR + f"rawData\\travel_{self.time}_normalized.edges"

        # interlayer network file path
        self.virtual_interlayer_file_path = ROOT_DIR + f"interlayer\\interlayer_v_{self.time}.edges"
        self.physical_interlayer_file_path = ROOT_DIR + f"interlayer\\interlayer_p_{self.time}.edges"

        # supramatrix file path
        self.supramatrix_file_path = ROOT_DIR + f"supraMatrix\\result_supra_matrix_{self.time}.csv"


class CONFIGPARAMS(object):

    city_num = 371
    cities = [i+1 for i in range(city_num)]

    # city distance dict -> dic_distance:{cityid: {cityid: distance}}
    def cityDistance():
        # city xy file path
        cityXY_filepath = ROOT_DIR + "rawData\\city_xy.csv"
        dic_distance = {}
        df = pd.read_csv(cityXY_filepath, header=None, index_col=None)
        dic_xy = {row[0]: (row[1], row[2]) for row in df.itertuples(index=False)}
        for start_city in dic_xy.keys():
            dic_distance[start_city] = {}
            for end_city in dic_xy.keys():
                x1, y1 = dic_xy[start_city]
                x2, y2 = dic_xy[end_city]
                distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                dic_distance[start_city][end_city] = distance

        return dic_distance

    # lambda_dict -> {p/v: {year: lambda}}
    def getlambda_dict():
        with open(ROOT_DIR + "bandwidth\\bandwidth.csv", "r") as f:
            reader = csv.reader(f)
            head = next(reader)
            lambda_dict = {"virtual": {}, "physical": {}}
            for row in reader:
                if row[0] == "virtual":
                    lambda_dict["virtual"][int(row[1])] = float(row[2])
                elif row[0] == "physical":
                    lambda_dict["physical"][int(row[1])] = float(row[2])

        return lambda_dict
    