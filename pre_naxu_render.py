import numpy as np
import cv2
import os
import datetime
import imageio

def main(path):

    world_timestamps = np.load(path + "/exports/000/world_timestamps.npy")

    cap = cv2.VideoCapture(path + "/exports/000/world.mp4")
    output_name = path + f"world_prenaxu.mp4"

    writer = imageio.get_writer(
        output_movie_path,
        fps=60,
        codec="libx264",
        bitrate="2000k"
    ) 
    
    frame_n = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(len(world_timestamps), length)
    
    while True:
        ret, oframe = cap.read()
        if not ret:
            break
        time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            
        oframe = cv2.putText(oframe, f'Video Time: {round(time, 2)} s', (50, 50),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness = 3, color=(150,150,150))
        oframe = cv2.putText(oframe, f'Pupil Time: {round(world_timestamps[frame_n], 2)} s', (50, 90),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness = 3, color=(150,150,150))
        oframe = cv2.putText(oframe, f'Frame: {frame_n}', (50, 130),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness = 3, color=(150,150,150))

        oframe = cv2.cvtColor(oframe, cv2.COLOR_RGB2BGR)
        writer.append_data(oframe)
        frame_n += 1

    writer.close()
        
rootdir = ""

for path in os.walk(rootdir):
        path = str(path[0]) + "/"
        if os.path.exists(path + "world_timestamps.npy") and os.path.exists(path + "user_info.csv") and os.path.exists(path + "/exports/000/world.mp4") and not os.path.exists(path + "world_prenaxu.mp4"):
            print(path)
            main(path)
