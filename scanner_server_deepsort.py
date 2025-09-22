# scanner_server_deepsort.py
# FastAPI server: /frame and /triangulate. Uses Ultralytics YOLO for detection and deep-sort-realtime (if available).
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import uvicorn, time, threading, math
import numpy as np, cv2
from ultralytics import YOLO
from collections import defaultdict
import os
from typing import List, Dict
# optional deep sort
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    HAVE_DEEPSORT = True
except Exception:
    HAVE_DEEPSORT = False

MODEL_NAME = "yolov8n.pt"  # change to your custom building detector if you have it
model = YOLO(MODEL_NAME)

app = FastAPI()
lock = threading.Lock()
if HAVE_DEEPSORT:
    tracker = DeepSort(max_age=30)
else:
    tracker = None

# In-memory tracks: track_id -> dict { observations: [ {...} ], last_seen_ts, bbox }
tracks = {}
next_track_id = 1
TRACK_MAX_MISSING = 6.0

def iou(boxA, boxB):
    xA=max(boxA[0], boxB[0]); yA=max(boxA[1], boxB[1])
    xB=min(boxA[2], boxB[2]); yB=min(boxA[3], boxB[3])
    interW=max(0,xB-xA); interH=max(0,yB-yA); inter=interW*interH
    aA=max(0,(boxA[2]-boxA[0]))*max(0,(boxA[3]-boxA[1]))
    aB=max(0,(boxB[2]-boxB[0]))*max(0,(boxB[3]-boxB[1]))
    denom=float(aA+aB-inter)
    if denom<=0: return 0.0
    return inter/denom

def associate_iou(dets, tracks_dict, thresh=0.3):
    assigned={}
    unassigned=set(range(len(dets)))
    for tid,t in tracks_dict.items():
        best_i=-1; best_score=0.0
        for i,d in enumerate(dets):
            if i not in unassigned: continue
            s=iou(t['bbox'], d)
            if s>best_score:
                best_score=s; best_i=i
        if best_i!=-1 and best_score>=thresh:
            assigned[best_i]=tid; unassigned.discard(best_i)
    return assigned, list(unassigned)

@app.post("/frame")
async def frame_endpoint(file: UploadFile = File(...),
                         timestamp: str = Form(None),
                         lat: str = Form(None),
                         lon: str = Form(None),
                         alt: str = Form(None),
                         yaw: str = Form(None),
                         pitch: str = Form(None),
                         roll: str = Form(None),
                         fx: str = Form(None),
                         fy: str = Form(None),
                         cx: str = Form(None),
                         cy: str = Form(None)):
    global next_track_id, tracks
    data = await file.read()
    arr = np.frombuffer(data, np.uint8); img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return JSONResponse({"error":"bad image"}, status_code=400)
    h,w = img.shape[:2]
    results = model.predict(img, imgsz=640, conf=0.25, verbose=False)
    dets = []
    for r in results:
        boxes = r.boxes
        if boxes is None: continue
        for b in boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0]); cls=int(b.cls[0])
            dets.append({'box':[x1,y1,x2,y2], 'score':conf, 'class':cls})
    now = time.time()
    response = []
    with lock:
        if HAVE_DEEPSORT:
            # format detections for deep sort [x1,y1,x2,y2,score, class]
            ds_dets = []
            for d in dets:
                bbox = d['box']; score = d['score']; cl = d['class']
                ds_dets.append([bbox[0], bbox[1], bbox[2], bbox[3], score, cl])
            tracks_out = tracker.update_tracks(ds_dets, frame=img)
            for tr in tracks_out:
                if not tr.is_confirmed(): continue
                tid = tr.track_id
                ltrb = tr.to_ltrb() if hasattr(tr, 'to_ltrb') else tr.to_ltrb()
                bbox = [int(ltrb[0]),int(ltrb[1]),int(ltrb[2]),int(ltrb[3])]
                tracks.setdefault(tid, {'bbox':bbox, 'last_seen_ts':now, 'observations':[]})
                obs = {'ts': timestamp, 'lat': float(lat) if lat else None, 'lon': float(lon) if lon else None, 'alt': float(alt) if alt else None,
                       'yaw': float(yaw) if yaw else 0.0, 'pitch': float(pitch) if pitch else 0.0, 'roll': float(roll) if roll else 0.0,
                       'box':bbox, 'img_w':w, 'img_h':h, 'fx':float(fx) if fx else 800.0, 'fy':float(fy) if fy else 800.0,
                       'cx':float(cx) if cx else w/2.0, 'cy':float(cy) if cy else h/2.0}
                tracks[tid]['observations'].append(obs)
                response.append({'box':bbox, 'score':1.0, 'class':0, 'track_id': str(tid)})
        else:
            # simple IOU tracker
            assigned, unassigned = associate_iou([d['box'] for d in dets], tracks)
            for di, tid in assigned.items():
                d = dets[di]; tracks[tid]['bbox'] = d['box']; tracks[tid]['last_seen_ts'] = now
                obs = {'ts': timestamp, 'lat': float(lat) if lat else None, 'lon': float(lon) if lon else None, 'alt': float(alt) if alt else None,
                       'yaw': float(yaw) if yaw else 0.0, 'pitch': float(pitch) if pitch else 0.0, 'roll': float(roll) if roll else 0.0,
                       'box':d['box'], 'img_w':w, 'img_h':h, 'fx':float(fx) if fx else 800.0, 'fy':float(fy) if fy else 800.0,
                       'cx':float(cx) if cx else w/2.0, 'cy':float(cy) if cy else h/2.0}
                tracks[tid].setdefault('observations', []).append(obs)
                response.append({'box': d['box'], 'score': d['score'], 'class': d['class'], 'track_id': str(tid)})
            for di in unassigned:
                d = dets[di]; tid = next_track_id; next_track_id += 1
                tracks[tid] = {'bbox': d['box'], 'last_seen_ts': now, 'observations': []}
                obs = {'ts': timestamp, 'lat': float(lat) if lat else None, 'lon': float(lon) if lon else None, 'alt': float(alt) if alt else None,
                       'yaw': float(yaw) if yaw else 0.0, 'pitch': float(pitch) if pitch else 0.0, 'roll': float(roll) if roll else 0.0,
                       'box': d['box'], 'img_w':w, 'img_h':h, 'fx':float(fx) if fx else 800.0, 'fy':float(fy) if fy else 800.0,
                       'cx':float(cx) if cx else w/2.0, 'cy':float(cy) if cy else h/2.0}
                tracks[tid]['observations'].append(obs)
                response.append({'box': d['box'], 'score': d['score'], 'class': d['class'], 'track_id': str(tid)})
        # cleanup
        dead = [tid for tid,t in tracks.items() if now - t['last_seen_ts'] > TRACK_MAX_MISSING]
        for tid in dead: del tracks[tid]
    return JSONResponse({"detections": response})

# ---- triangulation helpers (same idea as earlier) ----
from pyproj import CRS, Transformer
crs_geod = CRS.from_epsg(4979); crs_ecef = CRS.from_epsg(4978)
_ecef_transformer = Transformer.from_crs(crs_geod, crs_ecef, always_xy=True)
_ecef_to_geod = Transformer.from_crs(crs_ecef, crs_geod, always_xy=True)

def lla_to_ecef(lat, lon, alt):
    x,y,z = _ecef_transformer.transform(lon, lat, alt)
    return np.array([x,y,z])
def ecef_to_lla(x,y,z):
    lon, lat, alt = _ecef_to_geod.transform(x,y,z)
    return lat, lon, alt

def enu_from_ecef(points_ecef, ref_ecef, ref_lat, ref_lon):
    lat = math.radians(ref_lat); lon = math.radians(ref_lon)
    slat = math.sin(lat); clat=math.cos(lat); slon=math.sin(lon); clon=math.cos(lon)
    R = np.array([[-slon, clon, 0],
                  [-clon*slat, -slon*slat, clat],
                  [clon*clat, slon*clat, slat]])
    enu = (R @ (points_ecef - ref_ecef).T).T
    return enu

def ray_from_obs(obs):
    x1,y1,x2,y2 = obs['box'][0],obs['box'][1],obs['box'][2],obs['box'][3]
    u=(x1+x2)/2.0; v=(y1+y2)/2.0
    fx = obs.get('fx',800.0); fy = obs.get('fy',800.0); cx=obs.get('cx', obs.get('img_w',640)/2.0); cy=obs.get('cy', obs.get('img_h',480)/2.0)
    xn = (u - cx) / fx; yn = (v - cy) / fy
    cam_vec = np.array([xn, yn, 1.0]); cam_vec = cam_vec/np.linalg.norm(cam_vec)
    yaw = math.radians(obs.get('yaw',0.0)); pitch = math.radians(obs.get('pitch',0.0)); roll = math.radians(obs.get('roll',0.0))
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],[math.sin(yaw), math.cos(yaw),0],[0,0,1]])
    Ry = np.array([[math.cos(pitch),0,math.sin(pitch)],[0,1,0],[-math.sin(pitch),0,math.cos(pitch)]])
    Rx = np.array([[1,0,0],[0,math.cos(roll),-math.sin(roll)],[0,math.sin(roll),math.cos(roll)]])
    R = Rz @ Ry @ Rx
    world_vec = R @ cam_vec; world_vec = world_vec/np.linalg.norm(world_vec)
    return world_vec

def triangulate_track_obs(obs_list):
    if len(obs_list) < 2: return None
    ref = obs_list[0]
    if ref['lat'] is None or ref['lon'] is None: return None
    ref_ecef = lla_to_ecef(ref['lat'], ref['lon'], ref.get('alt',0.0))
    cam_ecefs = []; dirs = []
    for obs in obs_list:
        if obs['lat'] is None or obs['lon'] is None: continue
        cam_ecefs.append(lla_to_ecef(obs['lat'], obs['lon'], obs.get('alt',0.0)))
    if len(cam_ecefs) < 2: return None
    cam_ecefs = np.array(cam_ecefs)
    enu_positions = enu_from_ecef(cam_ecefs, ref_ecef, ref['lat'], ref['lon'])
    for obs in obs_list:
        if obs['lat'] is None or obs['lon'] is None: continue
        dirs.append(ray_from_obs(obs))
    P = enu_positions; D = np.array(dirs)
    A = np.zeros((3,3)); b = np.zeros(3)
    for i in range(len(P)):
        d = D[i].reshape(3,1)
        Iminus = np.eye(3) - (d @ d.T)
        A += Iminus; b += (Iminus @ P[i])
    try:
        x = np.linalg.solve(A,b)
    except Exception:
        x, *_ = np.linalg.lstsq(A,b, rcond=None)
    lat = ref['lat']; lon = ref['lon']
    latr = math.radians(lat); lonr = math.radians(lon)
    slat = math.sin(latr); clat = math.cos(latr); slon = math.sin(lonr); clon = math.cos(lonr)
    R = np.array([[-slon, clon, 0],[-clon*slat, -slon*slat, clat],[clon*clat, slon*clat, slat]])
    point_ecef = ref_ecef + (R.T @ x)
    lat_out, lon_out, alt_out = ecef_to_lla(point_ecef[0], point_ecef[1], point_ecef[2])
    residuals=[]
    for i in range(len(P)):
        v = (R.T @ (point_ecef - cam_ecefs[i]))
        v_norm = v / (np.linalg.norm(v)+1e-12)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v_norm, D[i]))))))
        residuals.append(ang)
    rms = float(np.sqrt(np.mean(np.array(residuals)**2))) if residuals else None
    return {'lat': float(lat_out), 'lon': float(lon_out), 'alt': float(alt_out), 'reproj_px_rms': rms}

@app.post("/triangulate")
async def triangulate(track_id: str = Form(...)):
    tid = int(track_id)
    with lock:
        if tid not in tracks: return JSONResponse({"error":"track not found"}, status_code=404)
        obs = tracks[tid].get('observations', [])
        valid = [o for o in obs if o.get('lat') is not None and o.get('lon') is not None]
        if len(valid) < 2: return JSONResponse({"error":"need at least 2 GPS-tagged observations"}, status_code=400)
    res = triangulate_track_obs(valid)
    if res is None: return JSONResponse({"error":"triangulation failed"}, status_code=500)
    return JSONResponse(res)

if __name__ == "__main__":
    uvicorn.run("scanner_server_deepsort:app", host="0.0.0.0", port=8000, log_level="info")
