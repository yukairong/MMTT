import json
import os

def parse_json_file(file_root):
    """
    解析json文件
    :param file_root:  JSON文件路径
    :return: JSON文件的数据内容
    """
    if not os.path.isfile(file_root) or not file_root.endswith(".json"):
        print(f"{file_root} is not a JSON File")

    file = None
    with open(file_root, encoding="utf-8") as f:
        file = json.load(f)
        # print(file)
    return file

def xyxy_convert_to_xywh(bbox, scale=(1,1)):
    """
    将xyxy坐标转换成xywh,左上角坐标，宽高
    :param bbox: (Xmin, Ymin, Xmax, Ymax)的坐标格式
    :param scale: (Width scale, Height scale) 放缩比例
    :return: (x, y, w, h) 左上角坐标,宽,高
    """
    x = bbox[0]
    y = bbox[1]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = x * scale[0]
    y = y * scale[1]
    w = w * scale[0]
    h = h * scale[1]

    return (x, y, w, h)


if __name__ == '__main__':
    annotations_root = r"D:\datasets\Wildtrack_dataset_full\Wildtrack_dataset\annotations_positions\00000000.json"
    parse_json_file(file_root=annotations_root)