"""
函数说明：
get_per_frame_from_dataset: 从dataset中传输图片模拟实车运行情况
get_per_frame_from_camera: 从双目读入图片
"""

import cv2
import os


def get_per_frame_from_dataset(file_name, img_index):
    jpgFile = os.listdir(file_name)
    jpgFile.sort()
    imgPath = os.path.join(file_name, jpgFile[img_index])
    frame = cv2.imread(imgPath)
    return frame, jpgFile[img_index]

