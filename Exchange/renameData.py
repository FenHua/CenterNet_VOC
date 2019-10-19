# coding utf-8
import random
import numpy as np
import os, shutil


# coco中图片按照image id顺序排序的，名称中只能包含数字所以需要统一命名
def renameData(xmlDir, imgDir):
    xmlFiles = os.listdir(xmlDir)             # xml文件夹目录
    xmlFiles.sort(key=lambda x: int(x[:-4]))  # 对xml文件进行排序
    total = len(xmlFiles)                     # 总共的xml文件
    cur = 0                                   # 计数器
    for xml in xmlFiles:
        cur += 1
        '''
        if cur % 500 == 1:
            print("Total/cur:", total, "/", cur)
        '''
        imgPath = imgDir + xml[:-4] + ".jpg"  # 对应的图片名称
        outName = ("%08d" % (cur))
        outXMLPath = ("%s/%s.xml" % (xmlDir, outName))  # 新的名称，以数字命名
        outImgPath = ("%s/%s.jpg" % (imgDir, outName))  # 以数字开始的命名
        # 将文件重新命名
        os.rename(xmlDir + xml, outXMLPath)
        os.rename(imgPath, outImgPath)
    print("picker number:", cur)     # 输出重新命名的文件数量


if __name__ == '__main__':
    xmlDir = "/home/yhq/Desktop/CenterNet/nwputest/annotations/"  # xml标注文件路经
    imgDir = "/home/yhq/Desktop/CenterNet/nwputest/images/"       # .jpg结尾的图片路经
    print(xmlDir)
    print(imgDir)
    renameData(xmlDir, imgDir)   # coco 格式的重新命名