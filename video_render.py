#!/usr/bin/env python3

import os
import json
import cv2
import numpy as np
import pandas as pd
import imageio
import datetime 

from scipy.interpolate import interp1d

FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
FPS          = 60

def build_color_map():
    color_map = {
        'q': (255,   0, 255),
        'w': (255, 192, 203),
        'e': ( 0, 204, 0),
        'r': (0,102, 0),
        't': (255, 0, 0),
        'y': (128, 0, 128),
        'u': (0, 0, 255),
        '1': (0, 255, 255),
        '2': (255,204,0),
        '3': (255,102,0),
        '0': (255, 255, 255),
        '7': (255, 255, 255),
        '8': (255, 255, 255),
        '9': (255, 255, 255),
    }

    for k, v in list(color_map.items()):
        color_map[f'{k}_launch'] = v
        color_map[f'{k}_land']   = v
    return color_map
    
def build_frame_times_map():
    frame_times = {
        'Default': (0,6),
        '0': (0,1),
        '7': (0,1),
        '8': (0,1),
        '9': (0,1),
        't': (0,1)
    }

    for k, v in list(frame_times.items()):
        frame_times[f'{k}_launch'] = v
        frame_times[f'{k}_land']   = v
    return frame_times

COLORS = build_color_map()
FRAME_DISPLAY_TIMES = build_frame_times_map()

MARKER_NAMES = {
    '0': 'T1 Begin',  '7': 'T6 End',   '8': 'T6 Begin', '9': 'T1 End',
    'r_land': 'TRACK_AHEAD', 'e_land': 'TRACK_EDGE',    'q_land': 'REFERENCE_POINT',
    'y_land': 'INSTRUMENTS', 'w_land': 'BRAKE_MARKER',     '1_land': '1st_APEX_FIXATION',
    '2_land': 'APEX_OKN',        '3_land': 'EXIT',          'u_land': 'LAF',
}

for k, v in list(MARKER_NAMES.items()):
    if k.endswith('_land'):
        MARKER_NAMES[k.replace('_land', '_launch')] = v


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
    
def get_display_frames(label):
    if label in FRAME_DISPLAY_TIMES:
        return FRAME_DISPLAY_TIMES[label]
    else:
        return FRAME_DISPLAY_TIMES['Default']

def get_end_frame_delta(label):
    return get_display_frames(label)[1]

def get_start_frame_delta(label):
    return get_display_frames(label)[0]

def read_markers(markers_json, video_timestamps):
    rows = []

    for frame in markers_json['markers']:
        frame_ts = frame['time']
        frame_markers, landing = [], None

        for m in frame['markers']:
            key, pos = m['name'], m['position']

            # skip unused
            if key in {'o', '5'}:
                continue

            frame_idx  = int(np.argmin(np.abs(frame_ts - video_timestamps)))
            frame_disp = frame_idx + get_start_frame_delta(key) # when frame appears on video, right now same as when annotated

            row = [frame_ts, key, float(pos[0]), float(pos[1]),
                   frame_idx, frame_disp]

            if key == '4':        
                landing = row
                continue

            rows.append(row)

            if key in {'q', 'w', 'e', 'r', 'y', 'u', '1', '2', '3'}:
                frame_markers.append(row)

        # assign launch/land
        if frame_markers and landing is not None:
            f_arr = np.asarray(frame_markers)
            landing_xy = np.asarray(landing[2:4], float)
            d = np.linalg.norm(f_arr[:, 2:4].astype(float) - landing_xy, axis=1)
            if len(d) == 2:  # exactly two markers in this block
                frame_markers[int(np.argmax(d))][1] += '_launch'
            frame_markers[int(np.argmin(d))][1] += '_land'

    rows.append([np.inf, '', 0.0, 0.0, 0, np.inf])
    rows.sort(key=lambda r: r[5])

    return rows

def parse_pairs(rows, max_gap):
    out, ended = [], True
    for i in range(len(rows)):
        if ended:
            ended = False
            continue
        if rows[i][0] < rows[i-1][0] + max_gap:
            ended = True
            rows[i-1][1] += '_start'
            rows[i][1] += '_end'
            out.append(list(rows[i-1]) + [rows[i][5], True])
        else:
            rows[i-1][1] += '_start_noEnd'
            out.append(list(rows[i-1]) + [rows[i-1][5] + 10, False])
            ended = False
    out.append([1e6, '', 0, 0, 0, int(1e6), int(1e6), False])  # sentinel
    return out

def build_blink_events(rows):
    return parse_pairs([r for r in rows if r[1] == 'p'], 0.4)

def build_signal_loss_events(rows):
    return parse_pairs([r for r in rows if r[1] == 'i'], 1000)


def put_time_overlay(img, t, frame):
    cv2.putText(img, f'Time: {t:.2f} s', (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, color = (150, 150, 150), fontScale=2, thickness = 4)
    cv2.putText(img, f'Frame: {frame}', (100, 150),
                cv2.FONT_HERSHEY_SIMPLEX, color = (150, 150, 150), fontScale=2, thickness = 4)

def put_text(img, txt, xy, col):
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, color= col, fontScale=2, thickness = 4)


def draw_marker(img, draw_dot, row, launch, land):
    key, x_n, y_n = row[1], row[2], row[3]
    col = COLORS.get(key, (0, 0, 0))[::-1]   # to BGR
    x_px, y_px = int(x_n * FRAME_WIDTH), int(y_n * FRAME_HEIGHT)

    if key in {'0', '7', '8', '9'}:
        put_text(img, MARKER_NAMES[key],
                 (int(0.5*FRAME_WIDTH), int(0.75*FRAME_HEIGHT)), col)
    elif key == 't':
        cv2.circle(img, (x_px, y_px), 5, col, -1)
        put_text(img, key, (x_px, y_px - 40), col)
    elif key in MARKER_NAMES:
        if key.endswith('_land'):
            cv2.circle(img, (x_px, y_px), 29, col, 2)
            put_text(img, MARKER_NAMES[key], (x_px, y_px - 40), col)
        if draw_dot:
            cv2.circle(img, (x_px, y_px), 5, col, -1)
                    
            if launch and land:
                p1 = (int(launch[2]*FRAME_WIDTH), int(launch[3]*FRAME_HEIGHT))
                p2 = (int(land[2]*FRAME_WIDTH),  int(land[3]*FRAME_HEIGHT))
                cv2.line(img, p1, p2, COLORS[land[1]][::-1], 1)


def render_video(video, markers,blinks, loss, out_path):
    cap   = cv2.VideoCapture(str(video))
                            
    vw = imageio.get_writer(
            str(out_path),
            fps=FPS,
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
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # activate new markers
        while frame_idx >= markers[m_idx][5]:
            r = markers[m_idx]
            m_idx += 1
            display_time = get_end_frame_delta(r[1])
            rows_out.append([r[4], r[0], t, frame_idx/FPS,
                             r[1], r[2], r[3]])
            active.append((display_time, frame_idx, r))

        # blink / signal-loss overlays
        if blinks[blink_idx][5] <= frame_idx <= blinks[blink_idx][6]:
            put_text(img, 'BLINK' + ('' if blinks[blink_idx][7] else ' (NO END)'),
                     (int(0.55*FRAME_WIDTH), int(0.75*FRAME_HEIGHT)), (150,150,150))
        elif frame_idx > blinks[blink_idx][6]:
            blink_idx += 1

        if loss[loss_idx][5] <= frame_idx <= loss[loss_idx][6]:
            put_text(img, 'SIGNAL LOSS',
                     (int(0.55*FRAME_WIDTH), int(0.75*FRAME_HEIGHT)), (150,150,150))
        elif frame_idx > loss[loss_idx][6]:
            loss_idx += 1

        # draw markers
        next_active, launch, land = [], None, None
        for ttl, fr, r in active:
            k = r[1]
            if k in {'p', 'p_start', 'p_end', 'p_start_noEnd', 'i', 'i_start', 'i_end', 'i_start_noEnd'}:
                continue
            if k.endswith('_launch'):
                launch = r
            elif k.endswith('_land'):
                land = r

            draw_dot = fr == frame_idx
            draw_marker(img, draw_dot, r, launch, land)
            if ttl > 1:
                next_active.append((ttl-1, fr, r))

        active = next_active

        put_time_overlay(img, t, frame_idx)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        vw.append_data(img)
        frame_idx += 1

    cap.release()
    vw.close()
    return rows_out

def main():

    vid_path = 'iZone Oulton Park 20250425 H264.mp4'
    json_path = 'iZone Oulton Park 20250425 H264.mp4-117.981-30823443 20250520.json'
    
    

    
    
    csv_folder = "naxu_csv"
    movie_folder = "postrender_naxu"
    
    os.makedirs(movie_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)
    
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    out_video = os.path.join(movie_folder, vid_path + f"_postender_naxu_{time}.mp4")
    output_csv = os.path.join(csv_folder, vid_path + f"_naxu_{time}.csv")

    try:
        gaze_path = 'gaze_positions.csv'
        ts_path = 'world_timestamps.csv'
        
        markers_json = load_json(json_path)
        gaze         = load_gaze_positions(gaze_path)
        world_ts     = np.genfromtxt(ts_path, delimiter=',')

        _ = interpolate_gaze(gaze, world_ts)
    except:
        print("No gaze data")

    frame_ts = get_frame_timestamps(vid_path,
                                    vid_path + ('_ts.npy'))
    markers  = read_markers(markers_json, frame_ts)
    blinks   = build_blink_events(markers)
    loss     = build_signal_loss_events(markers)

    rows = render_video(vid_path, markers, blinks, loss,
                        out_video)

    pd.DataFrame(
        rows,
        columns=['frame', 'timestamp', 'timestamp_alt',
                'annotation_label', 'x', 'y', 'naxu_timestamp'],
    ).to_csv(output_csv, index=False)


if __name__ == '__main__':
    main()
