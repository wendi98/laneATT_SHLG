import numpy as np


def magnetic_spikes_dict(file_path):
    magneticSpikesDict = {}
    fp = open(file_path, 'r')
    sample = fp.readlines()
    for line in sample:
        sample_ = line.split('    ')
        magneticSpikesDict[str(sample_[1])] = [sample_[3], sample_[4]]
    return magneticSpikesDict


def magnetic_spikes_yaw(mag_spikes_dict, img_name):
    return np.arctan((int(mag_spikes_dict[img_name][1]) - int(mag_spikes_dict[img_name][0])) / 7762)


def magnetic_spikes_offset(mag_spikes_dict, img_name):
    return mag_spikes_dict[img_name][0], mag_spikes_dict[img_name][1]


# if __name__ == '__main__':
    # magSpikesDict = magnetic_spikes_dict('C:/Users/14588/Desktop/synchronization/Test1.txt')
    # print(magnetic_spikes_yaw(magSpikesDict, '1641373070_666'))
