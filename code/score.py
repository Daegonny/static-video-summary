import os
import cv2
import numpy as np

class Video_frame:
    def __init__(self, id, path, frame, histogram):
        self.id = id
        self.path = path
        self.frame = frame
        self.histogram = histogram

def load_frames(path):
    video_frames = []
    file_names = os.listdir(path)
    counter = 0
    for file_name in file_names:
        frame = cv2.imread(path+file_name)
        hist = get_hsv_histrogram(frame)
        video_frame = Video_frame(counter, path+file_name, frame, hist)
        video_frames.append(video_frame)
        counter += 1
    return video_frames

def get_vector_normalized(v):
    norm = np.sqrt(np.sum([np.power(x,2) for x in v]))
    new_v = [x/norm for x in v]
    return new_v

def get_hsv_histrogram(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [16], [0,180])
    return hist[:,0]/(frame.shape[0]*frame.shape[1])

def compute_distance(f1, f2):
    # f1 = get_vector_normalized(f1)
    # f2 = get_vector_normalized(f2)
    return np.sum([np.absolute(x1-x2) for x1, x2 in zip(f1, f2)])

f_CUS_a = []
f_CUS_e = []
start = 21
size_videos = 50
size_users = 5
for video in range(size_videos):
    VSUMM1_frames = load_frames("../auto-summary/v"+str(video+start)+"/")
    for user in range (size_users):
        user1_frames = load_frames("../user-summary/v"+str(video+start)+"/user"+str(user+1)+"/")

        user1_rem = []
        VSUMM1_rem = []

        u_size = len(user1_frames)
        a_size = len(VSUMM1_frames)
        u_in = 0
        u_out = 0

        for frame1 in user1_frames:
            if frame1.id in user1_rem:
                pass
            else:
                for frame2 in VSUMM1_frames:
                    if frame2.id in VSUMM1_rem:
                        pass
                    else:
                        d = compute_distance(frame1.histogram, frame2.histogram)
                        if d < 0.5:
                            user1_rem.append(frame1.id)
                            VSUMM1_rem.append(frame2.id)
                            u_in += 1
                            break

        u_out = a_size - u_in
        CUS_a = u_in/u_size
        CUS_e = u_out/u_size

        f_CUS_a.append(CUS_a)
        f_CUS_e.append(CUS_e)

        print(video+start, user+1, CUS_a, CUS_e)

print(np.sum(f_CUS_a)/(size_users*size_videos), np.sum(f_CUS_e)/(size_users*size_videos))
