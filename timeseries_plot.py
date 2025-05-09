import csv 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

colors = {"q": (255, 0, 255), 'w': (255, 192, 203), "e": (0, 255, 0), "r": (0, 120, 0), "t": (255, 0, 0), "y": (255, 0, 255), "u": (0, 0, 255), "1": (0, 255, 255), "2": (255, 255, 0), "3": (255, 165, 0), "0": (255,255,255), "7": (255,255,255), "8": (255,255,255), "9": (255,255,255)}

for c in list(colors.keys()):
    colors[c + "_launch"] = colors[c]
    colors[c + "_land"] = colors[c]
    
    
data = pd.read_csv("naxu.csv")

data_to_plot = []

for frame, frame_data in data.groupby(by = 'frame'):
    land = None
    launch = None
    for _, point in frame_data.iterrows():
        if 'land' in point.marker_name:
                land = point
        elif "launch" in point.marker_name:
                launch = point
        else:
            data_to_plot.append([frame, frame_data.iloc[0].video_time, -1, (0,0,0), point.marker_name]) 
    if land is not None and launch is not None:
        dist = ((launch['x'] - land['x'])**2 + (launch['y'] - land['y'])**2)**0.5  
        marker = land.marker_name
        c = tuple(np.array(colors[marker])/255)
        data_to_plot.append([frame, frame_data.iloc[0].video_time, dist, c, marker]) 
    elif land is not None:
        #dist = 0.3 #((launch['x'] - land['x'])**2 + (launch['y'] - land['y'])**2)**0.5  
        marker = land.marker_name
        c = tuple(np.array(colors[marker])/255)
        data_to_plot.append([frame, frame_data.iloc[0].video_time, -1, c, marker])  
        

data_to_plot = pd.DataFrame(data_to_plot, columns = ['frame', 'time', 'ampl', 'color', 'marker'])  
segments = pd.read_csv("segment_ranges2.csv")       
#44AA00
n = 0
fig, ax = plt.subplots()

y_ticks = []
for _, segment in segments.iterrows():
    if "T6" in segment.seg: 
        continue
    if 'cool' in segment.seg:
        continue
    if 'out' in segment.seg:
        continue
    seg_data = data_to_plot[(data_to_plot['frame'] >= segment.start) & (data_to_plot['frame'] <= segment.end)]
    
    land_1 = seg_data[seg_data['marker'] == '1_land']
    t = land_1.iloc[0].time
    seg_data.time -= t
    y_ticks.append(segment.seg)



    for index in range(len(seg_data)):
        point = seg_data.iloc[index]
        print(point['marker'] )
        start_ts = point['time']
        if (index + 1) < len(seg_data):
            end_ts = seg_data.iloc[index+1]['time']
        else:
            end_ts = point['time']
        print(end_ts)
        plt.plot([start_ts,start_ts], [n, n], '.', color = point['color'])
        plt.plot([start_ts,end_ts], [n, n], color = point['color'])

            
    n -= 1
    
plt.yticks(np.arange(0,n,-1), y_ticks, fontsize=10)

plt.ylim(-8.5,0.5)
plt.xlim(-8.5, 8.5)
plt.xlabel("Time (s)")
#plt.ylabel("Amplitude")
plt.savefig(f"figures_with_labels/T1.svg")

plt.show()
