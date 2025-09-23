from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import JSONResponse
import uvicorn, time, threading, math
import numpy as np, cv2
from ultralytics import YOLO
from collections import defaultdict
import os, logging, json, pathlib
from typing import List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware

# Optional deep sort
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    HAVE_DEEPSORT = True
except Exception:
    HAVE_DEEPSORT = False

# ---------------- logging & persistence setup ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scanner_server")

PERSIST_OBS_FILE = pathlib.Path("observations.jsonl")
DETECTIONS_LOG_FILE = pathlib.Path("detections_log.jsonl")

def persist_obs(track_id, obs):
    """Append one observation for track_id to JSONL file."""
    try:
        entry = {"track_id": int(track_id), "obs": obs, "ts_saved": time.time()}
        with PERSIST_OBS_FILE.open("a", encoding="utf8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        logger.exception("Failed to persist observation")

def persist_detection_frame(frame_info):
    """Append frame-level detection info to DETECTIONS_LOG_FILE."""
    try:
        with DETECTIONS_LOG_FILE.open("a", encoding="utf8") as fh:
            fh.write(json.dumps(frame_info, default=str) + "\n")
    except Exception:
        logger.exception("Failed to persist detection frame")

# ---------------- FastAPI app & model init ----------------
MODEL_NAME = "yolov8n.pt"
model = YOLO(MODEL_NAME)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# request-logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("Incoming request: %s %s", request.method, request.url)
    resp = await call_next(request)
    logger.info("Completed request: %s %s -> %s", request.method, request.url, resp.status_code)
    return resp

lock = threading.Lock()
if HAVE_DEEPSORT:
    tracker = DeepSort(max_age=10)
    logger.info("DeepSort available: using deep-sort-realtime")
else:
    tracker = None
    logger.info("DeepSort not available: falling back to IOU tracker")

executor = ThreadPoolExecutor(max_workers=2)
async def run_inference(img):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: model.predict(img, imgsz=640, conf=0.25, verbose=False))

@app.get("/")
def root():
    return {"status":"ok","msg":"server alive"}

# ---------------- tracking state ----------------
tracks = {}           # tid -> {'bbox':..., 'last_seen_ts':..., 'observations':[...]}
next_track_id = 1
TRACK_MAX_MISSING = 6.0

# rehydrate observations from disk at startup
def rehydrate_tracks():
    global next_track_id, tracks
    if not PERSIST_OBS_FILE.exists():
        logger.info("No persistence file found; starting with empty tracks.")
        return
    try:
        max_tid = 0
        with PERSIST_OBS_FILE.open("r", encoding="utf8") as fh:
            for line in fh:
                try:
                    j = json.loads(line)
                    tid = int(j.get("track_id"))
                    obs = j.get("obs", {})
                    tracks.setdefault(tid, {'bbox': obs.get('box',[]), 'last_seen_ts': time.time(), 'observations': []})
                    tracks[tid]['observations'].append(obs)
                    max_tid = max(max_tid, tid)
                except Exception:
                    continue
        next_track_id = max(next_track_id, max_tid + 1)
        logger.info("Rehydrated %d tracks from %s; next_track_id=%d", len(tracks), PERSIST_OBS_FILE, next_track_id)
    except Exception:
        logger.exception("Failed to rehydrate observations")

# run rehydrate on import/start
rehydrate_tracks()

# ---------------- helper functions ----------------
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

# ---------------- /frame endpoint ----------------
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
    """
    Accept an image (multipart), run YOLO inference, run deep-sort (if available) or IOU tracker,
    record observations (persist them), and return detections with track_id values.
    """
    global next_track_id, tracks
    start_ts = time.time()
    data = await file.read()
    arr = np.frombuffer(data, np.uint8); img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Bad image in upload")
        return JSONResponse({"error":"bad image"}, status_code=400)
    h,w = img.shape[:2]

    # run inference in threadpool
    try:
        results = await run_inference(img)
    except Exception as e:
        logger.exception("Model inference failed")
        return JSONResponse({"error":"inference failed"}, status_code=500)

    dets = []
    for r in results:
        boxes = r.boxes
        if boxes is None: continue
        for b in boxes:
            try:
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0]); cls=int(b.cls[0])
                dets.append({'box':[x1,y1,x2,y2], 'score':conf, 'class':cls})
            except Exception:
                continue

    now = time.time()
    response = []
    frame_log_entry = {"frame_ts": start_ts, "received_ts": now, "num_detections": len(dets), "detections": []}

    with lock:
        if HAVE_DEEPSORT and tracker is not None:
            ds_dets = []
            for d in dets:
                bbox = d['box']; score = d['score']; cl = d['class']
                ds_dets.append([bbox[0], bbox[1], bbox[2], bbox[3], score, cl])
            try:
                tracks_out = tracker.update_tracks(ds_dets, frame=img)
            except Exception:
                logger.exception("DeepSort tracker.update_tracks failed")
                tracks_out = []
            for tr in tracks_out:
                try:
                    if not tr.is_confirmed(): continue
                    tid = int(tr.track_id)
                    ltrb = tr.to_ltrb() if hasattr(tr, 'to_ltrb') else tr.to_ltrb()
                    bbox = [int(ltrb[0]),int(ltrb[1]),int(ltrb[2]),int(ltrb[3])]
                    tracks.setdefault(tid, {'bbox':bbox, 'last_seen_ts':now, 'observations':[]})
                    obs = {'ts': timestamp, 'lat': float(lat) if lat else None, 'lon': float(lon) if lon else None,
                           'alt': float(alt) if alt else None, 'yaw': float(yaw) if yaw else 0.0, 'pitch': float(pitch) if pitch else 0.0,
                           'roll': float(roll) if roll else 0.0, 'box':bbox, 'img_w':w, 'img_h':h,
                           'fx':float(fx) if fx else 800.0, 'fy':float(fy) if fy else 800.0,
                           'cx':float(cx) if cx else w/2.0, 'cy':float(cy) if cy else h/2.0}
                    tracks[tid]['observations'].append(obs)
                    tracks[tid]['last_seen_ts'] = now
                    persist_obs(tid, obs)
                    response.append({'box':bbox, 'score':1.0, 'class':0, 'track_id': str(tid)})
                    frame_log_entry["detections"].append({"track_id": tid, "box": bbox})
                except Exception:
                    logger.exception("Error processing deep-sort track")
        else:
            # IOU-based associate
            assigned, unassigned = associate_iou([d['box'] for d in dets], tracks)
            for di, tid in assigned.items():
                try:
                    d = dets[di]; tracks[tid]['bbox'] = d['box']; tracks[tid]['last_seen_ts'] = now
                    obs = {'ts': timestamp, 'lat': float(lat) if lat else None, 'lon': float(lon) if lon else None,
                           'alt': float(alt) if alt else None, 'yaw': float(yaw) if yaw else 0.0, 'pitch': float(pitch) if pitch else 0.0,
                           'roll': float(roll) if roll else 0.0, 'box':d['box'], 'img_w':w, 'img_h':h,
                           'fx':float(fx) if fx else 800.0, 'fy':float(fy) if fy else 800.0,
                           'cx':float(cx) if cx else w/2.0, 'cy':float(cy) if cy else h/2.0}
                    tracks[tid].setdefault('observations', []).append(obs)
                    persist_obs(tid, obs)
                    response.append({'box': d['box'], 'score': d['score'], 'class': d['class'], 'track_id': str(tid)})
                    frame_log_entry["detections"].append({"track_id": tid, "box": d['box']})
                except Exception:
                    logger.exception("Error processing IOU assigned detection")
            for di in unassigned:
                try:
                    d = dets[di]; tid = next_track_id; next_track_id += 1
                    tracks[tid] = {'bbox': d['box'], 'last_seen_ts': now, 'observations': []}
                    obs = {'ts': timestamp, 'lat': float(lat) if lat else None, 'lon': float(lon) if lon else None,
                           'alt': float(alt) if alt else None, 'yaw': float(yaw) if yaw else 0.0, 'pitch': float(pitch) if pitch else 0.0,
                           'roll': float(roll) if roll else 0.0, 'box': d['box'], 'img_w':w, 'img_h':h,
                           'fx':float(fx) if fx else 800.0, 'fy':float(fy) if fy else 800.0,
                           'cx':float(cx) if cx else w/2.0, 'cy':float(cy) if cy else h/2.0}
                    tracks[tid]['observations'].append(obs)
                    persist_obs(tid, obs)
                    response.append({'box': d['box'], 'score': d['score'], 'class': d['class'], 'track_id': str(tid)})
                    frame_log_entry["detections"].append({"track_id": tid, "box": d['box']})
                except Exception:
                    logger.exception("Error creating new IOU track")
        # cleanup missing tracks
        dead = [tid for tid,t in tracks.items() if now - t['last_seen_ts'] > TRACK_MAX_MISSING]
        for tid in dead:
            try:
                del tracks[tid]
            except Exception:
                pass

    # persist frame-level log (non-blocking attempt)
    try:
        frame_log_entry["duration"] = time.time() - start_ts
        persist_detection_frame(frame_log_entry)
    except Exception:
        logger.exception("Failed to persist frame log")

    logger.info("Returning %d detections (frame took %.3fs)", len(response), time.time() - start_ts)
    return JSONResponse({"detections": response})

# ---------------- triangulation helpers ----------------
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

# ---------------- /triangulate endpoint ----------------
@app.post("/triangulate")
async def triangulate(track_id: str = Form(...)):
    tid = int(track_id)
    with lock:
        if tid not in tracks:
            logger.info("triangulate: track %s not found", tid)
            return JSONResponse({"error":"track not found"}, status_code=404)
        obs = tracks[tid].get('observations', [])
        valid = [o for o in obs if o.get('lat') is not None and o.get('lon') is not None]
        if len(valid) < 2:
            logger.info("triangulate: track %s has %d gps-tagged observations", tid, len(valid))
            return JSONResponse({"error":"need at least 2 GPS-tagged observations"}, status_code=400)
    res = triangulate_track_obs(valid)
    if res is None:
        logger.info("triangulate: failed for track %s", tid)
        return JSONResponse({"error":"triangulation failed"}, status_code=500)
    logger.info("triangulate: success for track %s -> lat=%s lon=%s rms=%s", tid, res.get('lat'), res.get('lon'), res.get('reproj_px_rms'))
    return JSONResponse(res)

# ---------------- script entrypoint ----------------
if __name__ == "__main__":
    # default local run (use Render start command externally to bind to $PORT)
    uvicorn.run("scanner_server_deepsort:app", host="0.0.0.0", port=8000, log_level="info")
