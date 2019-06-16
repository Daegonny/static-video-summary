import os
import cv2
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift

class Video_frame:
    def __init__(self, id, frame, feature, label):
        self.id = id
        self.frame = frame
        self.feature = feature
        self.label = label

def clean_folder(path):
    file_names = os.listdir(path)
    for file_name in file_names:
        os.remove(path+file_name)

def save_frames(video_frames, path):
    clean_folder(path)
    for video_frame in video_frames:
        cv2.imwrite(path + "Frame" + str(video_frame.id) + ".jpeg", video_frame.frame)

def get_feature2(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [16], [0,180])
    feature1 = hist[:,0]/(frame.shape[0]*frame.shape[1])
    return feature1

def get_vector_normalized(v):
    norm = np.sqrt(np.sum([np.power(x,2) for x in v]))
    new_v = [x/norm for x in v]
    return new_v

def compute_distance(f1, f2):
    return np.sum([np.absolute(x1-x2) for x1, x2 in zip(f1, f2)])


def get_video_frames(video_path, subsample = False):
    cap = cv2.VideoCapture(video_path)
    counter = 0
    frames = []
    if not cap.isOpened():
        raise IOError("Could not open video: " + video_path)

    while True:
        ret, frame = cap.read()

        if ret:
            if not subsample:
                frame = Video_frame(counter, frame, get_feature2(frame), None)
                frames.append(frame)
            elif counter % 30 == 1:
                frame = Video_frame(counter, frame, get_feature2(frame), None)
                frames.append(frame)
        else:
            break

        counter += 1

    cap.release()
    return frames


def get_key_frames(video_frames, diffs, t):
    key_frames = []
    counter = 0
    for diff in diffs:
        if diff > t:
            key_frames.append(video_frames[counter])
        counter += 1
    return key_frames

def get_diffs(video_frames):
    diffs = []
    l = len(video_frames)
    new_frames = []
    for i in range(l+1):
        if i == l-1:
            break
        distance = compute_distance(video_frames[i].feature,video_frames[i+1].feature)
        diffs.append(distance)

    return diffs

def set_label(video_frames, labels):
    for video_frame, label in zip(video_frames, labels):
        video_frame.label = label
    return video_frames

def remove_similar(video_frames, threshold):
    chosen_video_frames = []
    distances = []
    l = len(video_frames)

    for i in range(l-1):
        distance = compute_distance(video_frames[i].feature, video_frames[i+1].feature)
        if distance >=  threshold:
            chosen_video_frames.append(video_frames[i])
            if i == l-1:
                chosen_video_frames.append(video_frames[i+1])

    return chosen_video_frames

##########################################

start = 21
size_videos = 50

for video in range(size_videos):
    print("Video: "+str(video+start))
    video_frames = get_video_frames("../database/v"+str(video+start)+".mpg", True)
    diffs = get_diffs(video_frames)
    m_diffs = np.mean(diffs)
    s_diffs = np.std(diffs)
    t = m_diffs + s_diffs
    key_frames = get_key_frames(video_frames, diffs, t)
    key_frames = remove_similar(key_frames, t)
    save_frames(key_frames ,"../auto-summary/v"+str(video+start)+"/")
