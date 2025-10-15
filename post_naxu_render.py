#!/usr/bin/env python3

import sys
import os
import json
import cv2
import numpy as np
import pandas as pd
import imageio
import datetime 

from scipy.interpolate import interp1d

class Config:
    frame_width = 1280
    frame_height = 720
    fps = 60
        
    def __init__(self, annotations_json_path):
    
        annotation_info = load_json(annotations_json_path)

        self.color_map = self.get_color_map(annotation_info)
        self.frame_display_times = self.get_frame_times_map(annotation_info)
        self.gaze_class_markers = self.get_gaze_class_markers(annotation_info)
        self.skip_keys = annotation_info['SKIP_KEYS']
        self.blink = annotation_info['BLINK']
        self.signal_loss = annotation_info["SIGNAL_LOSS"]
        self.landing_point = annotation_info['LANDING_POINT']
        self.text_only_markers = annotation_info['TEXT_ONLY_MARKERS']
        self.point_only_markers = annotation_info['POINT_ONLY_MARKERS']
        
    def add_launch_land(self, dictionary):
        for k, v in list(dictionary.items()):
            dictionary[f'{k}_launch'] = v
            dictionary[f'{k}_land']   = v
        return dictionary

    def get_color_map(self, label_info):
        color_map = label_info['COLORMAP']
        color_map = self.add_launch_land(color_map)
        return color_map
        
    def get_frame_times_map(self, label_info):
        frame_times = label_info['FRAME_DISPLAY_TIMES']
        frame_times = self.add_launch_land(frame_times)
        return frame_times
        
    def get_gaze_class_markers(self, label_info):
        gaze_markers = label_info['GAZE_CLASS_MARKERS']
        gaze_markers = self.add_launch_land(gaze_markers)
        return gaze_markers

def load_json(path):
    with open(path) as fh:
        return json.load(fh)

def load_gaze_positions(csv_path):
    pupil = np.genfromtxt(csv_path, delimiter=',', names=True)
    return np.column_stack([
        pupil['gaze_timestamp'],
        pupil['norm_pos_x'],
        1 - pupil['norm_pos_y'],
        pupil['confidence'],
    ])


def interpolate_gaze(gaze, world_ts):
    f = interp1d(gaze[:, 0], gaze[:, 1:3], axis=0,
                  bounds_error=False, fill_value=np.nan)
    return f(world_ts)


def get_frame_timestamps(video, cache):
    folder = "./movie_timestamps"
    cache_path = os.path.join(folder, cache)

    if os.path.exists(cache_path):
        return np.load(cache_path)

    cap, times = cv2.VideoCapture(str(video)), []
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    cap.release()
    ts = np.asarray(times, float)
    if len(ts) > 0:
        os.makedirs(folder, exist_ok=True)
        np.save(cache_path, ts)
    return ts
    
def get_display_frames(cfg, label):
    if label in cfg.frame_display_times:
        return cfg.frame_display_times[label]
    else:
        return cfg.frame_display_times['Default']

def get_end_frame_delta(cfg, label):
    return get_display_frames(cfg, label)[1]

def get_start_frame_delta(cfg, label):
    return get_display_frames(cfg, label)[0]

def read_markers(cfg, markers_json, video_timestamps):
    rows = []

    for frame in markers_json['markers']:
        frame_ts = frame['time']
        frame_markers, landing = [], None

        for m in frame['markers']:
            key, pos = m['name'], m['position']

            # skip unused
            if key in cfg.skip_keys:
                continue

            frame_idx  = int(np.argmin(np.abs(frame_ts - video_timestamps)))
            frame_disp = frame_idx + get_start_frame_delta(cfg, key) # when frame appears on video, right now same as when annotated

            row = [frame_idx, frame_ts, frame_disp, key, float(pos[0]), float(pos[1])]

            if key == cfg.landing_point:        
                landing = row
                continue

            rows.append(row)

            if key in cfg.gaze_class_markers:
                frame_markers.append(row)

        # assign launch/land
        if frame_markers and landing is not None:
            f_arr = np.asarray(frame_markers)
            landing_xy = np.asarray(landing[4:], float)
            d = np.linalg.norm(f_arr[:, 4:].astype(float) - landing_xy, axis=1)
            if len(d) == 2:  # exactly two markers in this block
                frame_markers[int(np.argmax(d))][3] += '_launch'
            frame_markers[int(np.argmin(d))][3] += '_land'

    rows.sort(key=lambda r: r[2]) #sort by display time
    #rows.append([np.inf, '', 0.0, 0.0, 0, np.inf])

    marker_df = pd.DataFrame(rows, columns=["frame", "time", "display_frame", "label", "x", "y"])
    return marker_df

def parse_pairs(marker_df, mask, max_gap, max_show = 20):
    out = []
    ended = False
    idxs = marker_df[mask].index.tolist()
    for i in range(len(idxs)):
        start_i = idxs[i]
        if i < len(idxs) - 1:
            end_i = idxs[i+1]
        else:
            end_i = idxs[i] 

        if ended:
            ended = False
            continue
        if start_i != end_i and (marker_df.loc[start_i].time + max_gap > marker_df.loc[end_i].time):
            ended = True
            marker_df.at[start_i, "label"] += '_start'
            marker_df.at[end_i, "label"] += '_end'

            start_display_frame =  marker_df.loc[start_i].display_frame
            end_display_frame = marker_df.loc[end_i].display_frame
            out.append([start_display_frame, end_display_frame, True])
        else:
            marker_df.at[start_i, "label"] += '_start_noEnd'
            start_display_frame =  marker_df.loc[start_i].display_frame
            end_display_frame = start_display_frame + max_show
            out.append([start_display_frame, end_display_frame, False])
            
            ended = False

    out = pd.DataFrame(out, columns = ["start_frame", "end_frame", "paired"])
    return out

def build_blink_events(cfg, marker_df):
    blink_mask = marker_df.label == cfg.blink
    return parse_pairs(marker_df, blink_mask, 0.4)

def build_signal_loss_events(cfg, marker_df):
    signal_loss_mask = marker_df.label == cfg.signal_loss
    return parse_pairs(marker_df, signal_loss_mask, 1000, 240)

def put_time_overlay(img, t, frame):
    cv2.putText(img, f'Time: {t:.2f} s', (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, color = (150, 150, 150), fontScale=2, thickness = 4)
    cv2.putText(img, f'Frame: {frame}', (100, 150),
                cv2.FONT_HERSHEY_SIMPLEX, color = (150, 150, 150), fontScale=2, thickness = 4)

def put_text(img, txt, xy, col):
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, color= col, fontScale=2, thickness = 4)


def draw_marker(cfg, img, draw_dot, marker, launch, land):
    key, x_n, y_n = marker.label, marker.x, marker.y
    col = cfg.color_map.get(key, (0, 0, 0))[::-1]   # to BGR
    width, height = cfg.frame_width, cfg.frame_height
    
    x_px, y_px = int(x_n * width), int(y_n * height)

    if key in cfg.text_only_markers:
        put_text(img, cfg.text_only_markers[key],
                 (int(0.04*width), int(0.25*height)), col)
    if key in cfg.point_only_markers:
        cv2.circle(img, (x_px, y_px), 5, col, -1)
        put_text(img, key, (x_px, y_px - 40), col)
    if key in cfg.gaze_class_markers:
        if key.endswith('_land'):
            cv2.circle(img, (x_px, y_px), 29, col, 2)
            put_text(img, cfg.gaze_class_markers[key], (x_px, y_px - 40), col)
        if draw_dot:
            cv2.circle(img, (x_px, y_px), 5, col, -1)
                    
            if launch is not None and land is not None:
                p1 = (int(launch.x*width), int(launch.y*height))
                p2 = (int(land.x*width),  int(land.y*height))
                land_color = cfg.color_map.get(launch.label, (0, 0, 0))[::-1]   # to BGR
                cv2.line(img, p1, p2, land_color, 1)

def save_csv(rows, cols, out_path):
    pd.DataFrame(
        rows,
        columns = cols,
    ).to_csv(out_path, index=False)

def render_video(cfg, video, markers, blinks, signal_losses, out_path):
    cap   = cv2.VideoCapture(str(video))
                            
    vw = imageio.get_writer(
            str(out_path),
            fps=cfg.fps,
            codec="libx264",
            bitrate="2000k"
    ) 

    rows_out = []
    frame_idx = blink_idx = loss_idx = m_idx = 0
    active = []

    while True:
        ret, img = cap.read()
        if not ret:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        #put_text(img,"Label Position",(int(0.04*FRAME_WIDTH), int(0.25*FRAME_HEIGHT)), (255,255,255))

        # activate new markers
        while (m_idx < len(markers)) and (frame_idx >= markers.iloc[m_idx].display_frame):
            marker = markers.iloc[m_idx]
            m_idx += 1
            display_time = get_end_frame_delta(cfg, marker.label)
            rows_out.append([marker.frame, marker.time, ts, frame_idx/cfg.fps,
                             marker.label, marker.x, marker.y])
            active.append((display_time, frame_idx, marker))

        # blink / signal-loss overlays
        if blink_idx < len(blinks):
            blink = blinks.iloc[blink_idx]
            if blink.start_frame <= frame_idx <= blink.end_frame:
                put_text(img, 'BLINK' + ('' if blink.paired else ' (NO END)'),
                        (int(0.55*cfg.frame_width), int(0.75*cfg.frame_height)), (150,150,150))
            elif frame_idx > blink.end_frame:
                blink_idx += 1
        if loss_idx < len(signal_losses):
            loss = signal_losses.iloc[loss_idx]
            if loss.start_frame <= frame_idx <= loss.end_frame:
                put_text(img, 'SIGNAL LOSS'  + ('' if loss.paired else ' (NO END)'),
                        (int(0.40*cfg.frame_width), int(0.75*cfg.frame_height)), (150,150,150))
            elif frame_idx > loss.end_frame:
                loss_idx += 1

        # draw markers
        next_active, launch, land = [], None, None
        for frames_to_live, marker_frame, marker in active:
            key = marker.label
            if key[0] in [cfg.blink, cfg.signal_loss]:
                continue
            if key.endswith('_launch'):
                launch = marker
            elif key.endswith('_land'):
                land = marker

            draw_dot = marker_frame == frame_idx
            draw_marker(cfg, img, draw_dot, marker, launch, land)
            if frames_to_live > 1:
                next_active.append((frames_to_live-1, marker_frame, marker))

        active = next_active

        #put_time_overlay(img, t, frame_idx)
                
        #cv2.imshow("Test", img)
        #cv2.waitKey(1)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        vw.append_data(img)
        frame_idx += 1

    cap.release()
    vw.close()
    return rows_out

def main(video_path, naxu_json_path, config_json_path):
    
    cfg = Config(config_json_path)
    
    csv_folder = "naxu_csv"
    movie_folder = "postnaxu_render"
    
    os.makedirs(movie_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)
    
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_video = os.path.join(movie_folder, video_path + f"_postnaxu_render_{time}.mp4")
    output_csv = os.path.join(csv_folder, video_path + f"_naxu_{time}.csv")

    try:
        gaze_path = 'gaze_positions.csv'
        ts_path = 'world_timestamps.csv'
        gaze         = load_gaze_positions(gaze_path)
        world_ts     = np.genfromtxt(ts_path, delimiter=',')

        _ = interpolate_gaze(gaze, world_ts)
    except:
        print("No gaze data")

    frame_ts = get_frame_timestamps(video_path,
                                    video_path + ('_ts.npy'))
                                    
    markers_json = load_json(naxu_json_path)
    markers  = read_markers(cfg, markers_json, frame_ts)
    blinks   = build_blink_events(cfg, markers)
    loss     = build_signal_loss_events(cfg, markers)

    rows = render_video(cfg, video_path, markers, blinks, loss,
                        out_video)
    csv_columns = ['frame', 'naxu_timestamp', 'timestamp', 'timestamp_alt', 
                'annotation_label', 'x', 'y']
    save_csv(rows, csv_columns, output_csv)

if __name__ == '__main__':

    vid_path = 'test.mp4'
    json_path =  'naxu_annotations_for_test.json'
    config_path = 'annotations_info.json'

    if sys.argv and len(sys.argv) >= 3:
        vid_path = sys.argv[1]
        json_path = sys.argv[2]
    if sys.argv and len(sys.argv) >= 4:
        config_path = sys.argv[3]
    
    main(vid_path, json_path, config_path)
