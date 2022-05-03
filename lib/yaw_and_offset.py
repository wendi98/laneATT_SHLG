"""
获取车辆中心与车道线中心的偏差
使用ipm,计算偏航和偏移
"""

import glob
import os
import cv2
import numpy as np
import joblib
from numpy import polyfit
from tqdm import tqdm

from magnetic_spikes import *

calibrationColor = (0, 0, 255)
laneColor = (255, 0, 0)
centerColor = (0, 255, 0)
selectCenterColor = (0, 255, 255)
selectLaneColor = (255, 0, 255)
intersectColor = (192, 220, 0)


def get_matrix(src_point, dst_point):
    matrix = cv2.getPerspectiveTransform(src_point, dst_point)
    matrix = cv2.invert(matrix, matrix)
    matrix = matrix[1]

    return matrix


# 特定点的ipm变换，src->dst
def point_ipm(points, matrix):
    result = []
    for point in points:
        xDst = ((matrix[1][1] - matrix[2][1] * point[1]) * (matrix[2][2] * point[0] - matrix[0][2]) -
                (matrix[0][1] - matrix[2][1] * point[0]) * (matrix[2][2] * point[1] - matrix[1][2])) / \
               ((matrix[1][1] - matrix[2][1] * point[1]) * (matrix[0][0] - matrix[2][0] * point[0]) -
                (matrix[0][1] - matrix[2][1] * point[0]) * (matrix[1][0] - matrix[2][0] * point[1]))
        yDst = ((matrix[1][0] - matrix[2][0] * point[1]) * (matrix[2][2] * point[0] - matrix[0][2]) -
                (matrix[0][0] - matrix[2][0] * point[0]) * (matrix[2][2] * point[1] - matrix[1][2])) / \
               ((matrix[1][0] - matrix[2][0] * point[1]) * (matrix[0][1] - matrix[2][1] * point[0]) -
                (matrix[0][0] - matrix[2][0] * point[0]) * (matrix[1][1] - matrix[2][1] * point[1]))
        result.append([round(xDst), round(yDst)])
    return result


# 车道线ipm变换
def lane_ipm(points_dict, matrix):
    resultIpm = []
    if points_dict.__len__() != 2:
        return
    for i in range(points_dict.__len__()):
        points = points_dict[str(i)]
        res = point_ipm(points, matrix)
        resultIpm.append(res)

    return resultIpm


# 获取ipm变换后的标定线
def calibration_lane_ipm(calibration_dict, matrix):
    resultIpm = {}

    items = ['trainCenterCoefPoints', 'tenMeterCoefPoints', 'fifteenMeterCoefPoints', 'twentyMeterCoefPoints']
    for item in items:
        points = calibration_dict[item]
        res = point_ipm(points, matrix)
        resultIpm[item] = res

    return resultIpm


# 获取车道线中心
def lane_center(result_ipm):
    result = []

    if len(result_ipm) != 2:
        return result

    for i in range(len(result_ipm)):
        result_ipm[i] = sorted(result_ipm[i], key=lambda point: point[1])

    pointsList = [min(result_ipm, key=len)]
    for i in range(len(pointsList[0])):
        x = round((result_ipm[0][i][0] + result_ipm[1][i][0]) / 2)
        y = round((result_ipm[0][i][1] + result_ipm[1][i][1]) / 2)
        result.append([x, y])

    return result


# 相机标定信息转化
def calibration_lane(train_center, ten_meter_points, fifteen_meter_points, twenty_meter_points):
    calibrationDict = {}

    calibrationDict['trainCenterCoef'] = polyfit(train_center[:, 1], train_center[:, 0], 1)
    calibrationDict['tenMeterCoef'] = polyfit(ten_meter_points[:, 0], ten_meter_points[:, 1], 1)
    calibrationDict['fifteenMeterCoef'] = polyfit(fifteen_meter_points[:, 0], fifteen_meter_points[:, 1], 1)
    calibrationDict['twentyMeterCoef'] = polyfit(twenty_meter_points[:, 0], twenty_meter_points[:, 1], 1)

    xPoint1 = round(calibrationDict['trainCenterCoef'][0] * 200 + calibrationDict['trainCenterCoef'][1])
    xPoint2 = round(calibrationDict['trainCenterCoef'][0] * 359 + calibrationDict['trainCenterCoef'][1])
    calibrationDict['trainCenterCoefPoints'] = [[xPoint1, 200], [xPoint2, 359]]

    items = ['tenMeterCoef', 'fifteenMeterCoef', 'twentyMeterCoef']
    for item in items:
        yPoint1 = round(calibrationDict[item][0] * 0 + calibrationDict[item][1])
        yPoint2 = round(calibrationDict[item][0] * 639 + calibrationDict[item][1])
        points = [[0, yPoint1], [639, yPoint2]]
        calibrationDict[item + 'Points'] = points

    return calibrationDict


def draw_points(draw_img, points, color):
    for point in points:
        cv2.circle(draw_img, point, 2, color, 2)

    return draw_img


def draw_calibration_lane(draw_img, calibration_dict, color):
    items = ['trainCenterCoefPoints', 'tenMeterCoefPoints', 'fifteenMeterCoefPoints', 'twentyMeterCoefPoints']
    for item in items:
        draw_img = cv2.line(draw_img,
                            tuple(calibration_dict[item][0]),
                            tuple(calibration_dict[item][1]),
                            color=color,
                            thickness=1)
    return draw_img


# 得到参与计算偏移与偏航的车道线点的上下界
def boundary(calibration_dict, matrix):
    calibPointsDict = calibration_lane_ipm(calibration_dict, matrix)
    up = calibPointsDict['tenMeterCoefPoints'][0][1]
    down = calibPointsDict['twentyMeterCoefPoints'][0][1]

    return [up, down]


# 得到参与计算偏航的车道线点
def select_points(calibration_points_ipm_dict, lane_center_points):
    up = calibration_points_ipm_dict['tenMeterCoefPoints'][0][1]
    down = calibration_points_ipm_dict['twentyMeterCoefPoints'][0][1]
    selectivePoints = [p for p in lane_center_points if p[1] <= up]
    selectivePoints = [p for p in selectivePoints if p[1] >= down]

    return selectivePoints


def calibration_lane_ipm_coef(calibration_points_ipm_dict):
    trainCenterPoints = np.array(calibration_points_ipm_dict['trainCenterCoefPoints'])
    calibration_points_ipm_dict['trainCenterCoef'] = polyfit(trainCenterPoints[:, 1], trainCenterPoints[:, 0], 1)

    items = ['tenMeterCoef', 'fifteenMeterCoef', 'twentyMeterCoef']
    for item in items:
        points = np.array(calibration_points_ipm_dict[item + 'Points'])
        calibration_points_ipm_dict[item] = polyfit(points[:, 0], points[:, 1], 1)

    return calibration_points_ipm_dict


# 计算两直线交点
def line_intersection(lane1_coef, lane2_coef):
    D = lane1_coef[0] * lane2_coef[1] - lane2_coef[0] * lane1_coef[1]
    x = (lane1_coef[1] * lane2_coef[2] - lane2_coef[1] * lane1_coef[2]) / D
    y = (lane2_coef[0] * lane1_coef[2] - lane1_coef[0] * lane2_coef[2]) / D

    return [round(x), round(y)]


# 得到参与计算偏移的车道线点
def key_points(intersect_dict, selective_lane_coef, selective_center_coef, calibration_lane_coef):
    selectiveCenterCoef = [1, -selective_center_coef[0], -selective_center_coef[1]]
    intersect_dict['laneCenter'] = line_intersection(selectiveCenterCoef, calibration_lane_coef)

    for i in range(len(selective_lane_coef)):
        laneCoef = [1, -selective_lane_coef[i][0], -selective_lane_coef[i][1]]
        intersect_dict['lane' + str(i)] = line_intersection(laneCoef, calibration_lane_coef)

    return intersect_dict


# 计算偏移和偏航，偏航：车辆向右偏为负，偏移：车辆向右偏为负
def yaw_and_bias(calibration_points_ipm_dict, intersect_dict, selective_points, selective_lane):
    trainCenterCoef = calibration_points_ipm_dict['trainCenterCoef']
    selective_points = np.array(selective_points)
    selectivePointsCoef = polyfit(selective_points[:, 1], selective_points[:, 0], 1)

    yaw = np.arctan(
        (trainCenterCoef[0] - selectivePointsCoef[0]) / (1 + trainCenterCoef[0] * selectivePointsCoef[0]))

    selectiveLaneCoef = []
    for lane in selective_lane:
        lane = np.array(lane)
        selectiveLaneCoef.append(polyfit(lane[:, 1], lane[:, 0], 1))

    items = ['fifteenMeterCoef']
    for item in items:
        calibrationLaneCoef = [calibration_points_ipm_dict[item][0], -1, calibration_points_ipm_dict[item][1]]
        intersect_dict = key_points(intersect_dict, selectiveLaneCoef, selectivePointsCoef, calibrationLaneCoef)

    laneDistance = np.linalg.norm(np.array(intersect_dict['lane0']) - np.array(intersect_dict['lane1']))
    centerDistance = np.linalg.norm(np.array(intersect_dict['trainCenter']) - np.array(intersect_dict['laneCenter']))
    bias = centerDistance * 3.5 / laneDistance
    if intersect_dict['trainCenter'][0] < intersect_dict['laneCenter'][0]:
        bias = -bias

    return yaw, bias


# ipm结果可视化
def visual_result(src_img, points_dict, matrix, calibration_dict, intersect_dict, view=False):
    # 得到车道线与车道中心线的ipm
    laneIpm = lane_ipm(points_dict, matrix)
    if not laneIpm:
        return 0, 0
    laneCenter = lane_center(laneIpm)

    # 选定的车道线与车道线中心点
    selectivePoints = laneCenter
    # selectivePoints = select_points(calibration_dict, laneCenter)
    selectiveLane = []
    for lane in laneIpm:
        selectiveLanePoints = select_points(calibration_dict, lane)
        selectiveLane.append(selectiveLanePoints)

    # 计算偏移和偏航
    camYaw, camBias = yaw_and_bias(calibration_dict, intersect_dict, selectivePoints, selectiveLane)

    # 可视化
    if view:
        r, c = src_img.shape[:2]
        resultImg = cv2.warpPerspective(src_img, matrix, (c, r), flags=cv2.WARP_INVERSE_MAP,
                                        borderMode=cv2.BORDER_CONSTANT)
        cv2.putText(resultImg, 'camYaw: ' + str(camYaw), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(resultImg, 'offset: ' + str(camBias), (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # 原图车道线检测结果
        for i in range(points_dict.__len__()):
            points = points_dict[str(i)]
            for curr_p, next_p in zip(points[:-1], points[1:]):
                src_img = cv2.line(src_img,
                                  tuple(curr_p),
                                  tuple(next_p),
                                  color=laneColor,
                                  thickness=3)

        # 标定线
        resultImg = draw_calibration_lane(resultImg, calibration_dict, calibrationColor)
        # srcImg = draw_points(srcImg, pointSrc, calibrationColor)
        # resultImg = draw_points(resultImg, pointDest, calibrationColor)

        # 车道线中心
        if laneCenter:
            resultImg = draw_points(resultImg, laneCenter, centerColor)

        # 车道线
        for i in range(len(laneIpm)):
            resultImg = draw_points(resultImg, laneIpm[i], laneColor)
        for lane in selectiveLane:
            resultImg = draw_points(resultImg, lane, selectCenterColor)

        # 交点
        for value in intersect_dict.values():
            resultImg = draw_points(resultImg, [value], intersectColor)
        resultImg = cv2.line(resultImg,
                             tuple(intersect_dict['trainCenter']),
                             tuple(intersect_dict['laneCenter']),
                             color=intersectColor,
                             thickness=2)
        imgs = np.hstack([src_img, resultImg])
        # cv2.imshow("results", imgs)
        # cv2.waitKey(30)
        return imgs, camYaw, camBias
    return camYaw, camBias


# if __name__ == "__main__":
def post_processing(img, points_dict, img_name, matrix_ipm, calib_points_dict, intersect_dict):
    # # ipm
    # srcPoints = joblib.load(os.path.join('pkl_files', 'ipm_src.pkl'))
    # destPoints = joblib.load(os.path.join('pkl_files', 'ipm_dest.pkl'))
    # matrixIpm = get_matrix(srcPoints, destPoints)
    #
    # # calibration
    # trainCenter = joblib.load(os.path.join('pkl_files', 'center.pkl'))
    # tenMeterPoints = joblib.load(os.path.join('pkl_files', 'tenMeter.pkl'))
    # fifteenMeterPoints = joblib.load(os.path.join('pkl_files', 'fifteenMeter.pkl'))
    # twentyMeterPoints = joblib.load(os.path.join('pkl_files', 'twentyMeter.pkl'))
    #
    # calibDict = calibration_lane(trainCenter, tenMeterPoints, fifteenMeterPoints, twentyMeterPoints)
    # calibPointsDict = calibration_lane_ipm(calibDict, matrixIpm)
    # calibPointsDict = calibration_lane_ipm_coef(calibPointsDict)
    #
    # # intersection
    # intersectDict = {}
    # trainCenterLineCoef = [1, -calibPointsDict['trainCenterCoef'][0], -calibPointsDict['trainCenterCoef'][1]]
    # fifteenMeterLineCoef = [calibPointsDict['fifteenMeterCoef'][0], -1, calibPointsDict['fifteenMeterCoef'][1]]
    # intersectDict['trainCenter'] = line_intersection(trainCenterLineCoef, fifteenMeterLineCoef)

    # save image and txt
    camYaw, camBias = visual_result(img, points_dict, matrix_ipm, calib_points_dict, intersect_dict)
    with open('C:/Users/14588/Desktop/yawdata.txt', 'a+') as f:
        f.write(img_name)
        f.write('\t')
        f.write(str(camYaw))
        f.write('\t')
        f.write(str(camBias))
        f.write('\t')
    return camYaw, camBias
