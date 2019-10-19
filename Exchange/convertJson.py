# coding:utf-8
# 将VOC xml转换为coco json格式
import os
import json
import xml.etree.ElementTree as ET

START_BOUNDING_BOX_ID = 1

# 注意下面的dict存储的是实际检测的类别，需要根据自己的实际数据进行修改
# 注意类别名称和xml文件中的标注名称一致
PRE_DEFINE_CATEGORIES = {"aircraft": 1}


# 从root目录下获取名称为name的变量
def get(root, name):
    vars = root.findall(name)  # 返回所有名称为name的变量
    return vars


# 从root目录下获取名称为name的变量，并检查相应变量的数量
def get_and_check(root, name, length):
    vars = root.findall(name)   # 获取所有name变量
    if len(vars) == 0:
        # 没有发现此变量
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        # 与设定的长度不一致
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


# 获取文件名称并转换为int型
def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]   # 获取文件名（数字字符串）
        return int(filename)    # 转换为int型
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.' % (filename))


# 将xml文件转换为json格式
def convert(xml_dir, json_file):
    xmlFiles = os.listdir(xml_dir)      # xml文件目录
    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}      # json格式
    categories = PRE_DEFINE_CATEGORIES  # 类别
    bnd_id = START_BOUNDING_BOX_ID      # 类别对应bbox的id
    num = 0   # 记录读取的xml文件数目
    # 迭代读取xml目录中的文件
    for line in xmlFiles:
        num += 1
        '''
        if num % 50 == 0:
            print("processing ", num, "; file ", line)  # 每读取50个xml文件显示一次
        '''
        xml_f = os.path.join(xml_dir, line)             # xml文件路经
        tree = ET.parse(xml_f)                          # xml文件解析
        root = tree.getroot()                           # 获取xml文件的根
        filename = line[:-4]                            # 文件名必须是一个数字
        image_id = get_filename_as_int(filename)        # 将文件名转换为int型
        size = get_and_check(root, 'size', 1)           # 检查文件名
        width = int(get_and_check(size, 'width', 1).text)    # 获取当前图片的宽
        height = int(get_and_check(size, 'height', 1).text)  # 获取当前图片的高
        image = {'file_name': (filename + '.jpg'), 'height': height, 'width': width,
                 'id': image_id}                        # image的元组形式
        json_dict['images'].append(image)               # 写入json文件
        # 获取当前图片中的所需类别的标注信息
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text   # 获取当前的类别名称
            if category not in categories:
                # 如果此类别不在当前类别目录，则将此新的类别加入
                new_id = len(categories)
                categories[category] = new_id   # id
            category_id = categories[category]  # id
            bndbox = get_and_check(obj, 'bndbox', 1)    # 获取bbx
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert (xmax > xmin)
            assert (ymax > ymin)
            o_width = abs(xmax - xmin)    # bbx的宽
            o_height = abs(ymax - ymin)   # bbx的高
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}    # 标注信息的元组
            json_dict['annotations'].append(ann)    # 将标注信息加入json文件
            bnd_id = bnd_id + 1
    # json文件大类别信息
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    # json文件的写入
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


'''
在生成coco格式的annotations文件之前需要进行以下两步:
1.执行renameData.py对xml和jpg统一命名；
2.执行splitData方法，切分好对应的train/val/test数据集
'''
if __name__ == '__main__':
    #folder_list=["test"]
    folder_list = ["train", "val", "test"]  # 对应三个实验类别的文件夹
    base_dir = "/home/yhq/Desktop/CenterNet/nwputest/annotations/"  # 本地实际标注文件路径
    for i in range(1):
        folderName = folder_list[i]
        xml_dir = base_dir + folderName + "/annotations/"
        json_dir = base_dir + folderName + "/instances_" + folderName + ".json"
        print("deal: ", folderName)
        print("xml dir: ", xml_dir)
        print("json file: ", json_dir)
        convert(xml_dir, json_dir)