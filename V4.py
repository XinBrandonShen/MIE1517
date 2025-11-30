import cv2
import numpy as np
from ultralytics import YOLO

WEIGHTS = r"C:\Users\Owner\Desktop\MIE1517\project\runs\segment\tableware_yolov11m\weights\best.pt"

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(s)],      # TL
        pts[np.argmin(d)],      # TR
        pts[np.argmax(s)],      # BR
        pts[np.argmax(d)],      # BL
    ], dtype=np.float32)

def fill_holes(mask255):
    h, w = mask255.shape[:2]
    flood = mask255.copy()
    ffmask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, ffmask, (0,0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask255, holes)

def clean_mask(mask01):
    m = (mask01 > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    # keep largest CC
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = (lab == largest).astype(np.uint8) * 255
    m = fill_holes(m)
    return m

def get_table_mask_yolo(model, img_bgr, table_name_contains="table"):
    res = model.predict(img_bgr, imgsz=960, conf=0.25, verbose=False)[0]
    if res.masks is None: 
        return None
    names = res.names
    cls = res.boxes.cls.cpu().numpy().astype(int)
    masks = res.masks.data.cpu().numpy()  # N x Hm x Wm

    table_ids = [i for i,n in names.items() if table_name_contains.lower() in str(n).lower()]
    cand = []
    for k in range(len(masks)):
        if (not table_ids) or (cls[k] in table_ids):
            cand.append((masks[k].sum(), k))
    if not cand:
        return None

    _, kbest = max(cand, key=lambda x: x[0])
    m = (masks[kbest] > 0.5).astype(np.uint8) * 255
    m = cv2.resize(m, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return m

def warp_from_corners(img, corners):
    corners = order_points(corners)
    tl, tr, br, bl = corners
    W = int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))
    H = int(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))
    W, H = max(W, 10), max(H, 10)
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (W,H)), M

def lsd_table_corners(img_bgr, mask255):
    # boundary band
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    er = cv2.erode(mask255, k, 1)
    boundary = cv2.subtract(mask255, er)
    band = cv2.dilate(boundary, k, 2)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return None

    # keep lines whose midpoint lies in band
    kept = []
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        mx, my = int(0.5*(x1+x2)), int(0.5*(y1+y2))
        if 0 <= mx < band.shape[1] and 0 <= my < band.shape[0] and band[my,mx] > 0:
            kept.append([x1,y1,x2,y2])
    if len(kept) < 6:
        return None

    kept = np.array(kept, dtype=np.float32)

    # cluster by direction using (cos2θ, sin2θ) to remove 180° ambiguity
    dirs = kept[:,2:4] - kept[:,0:2]
    ang = np.arctan2(dirs[:,1], dirs[:,0])
    feat = np.stack([np.cos(2*ang), np.sin(2*ang)], axis=1)

    # simple 2-means (no sklearn): pick farthest pair init, iterate
    c1 = feat[0]
    c2 = feat[np.argmax(np.linalg.norm(feat - c1, axis=1))]
    for _ in range(15):
        d1 = np.linalg.norm(feat - c1, axis=1)
        d2 = np.linalg.norm(feat - c2, axis=1)
        lab = (d2 < d1).astype(np.int32)
        if np.any(lab==0): c1 = feat[lab==0].mean(axis=0)
        if np.any(lab==1): c2 = feat[lab==1].mean(axis=0)

    # For each cluster, find 2 “extreme” lines by projecting midpoints onto the normal
    corners = []
    groups = [kept[lab==0], kept[lab==1]]
    for g in groups:
        if len(g) < 2:
            return None
        mid = 0.5*(g[:,0:2] + g[:,2:4])
        d = g[:,2:4] - g[:,0:2]
        n = np.stack([d[:,1], -d[:,0]], axis=1)
        n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        s = np.sum(mid * n, axis=1)  # signed offset proxy
        i_min, i_max = np.argmin(s), np.argmax(s)
        corners.append(g[i_min]); corners.append(g[i_max])

    # Convert 4 representative segments -> 4 infinite lines; then intersect adjacent pairs.
    # line through two points p1, p2: l = p1 x p2
    def seg_to_line(seg):
        x1,y1,x2,y2 = seg
        p1 = np.array([x1,y1,1.0])
        p2 = np.array([x2,y2,1.0])
        l = np.cross(p1,p2)
        l /= (np.linalg.norm(l[:2]) + 1e-9)
        return l

    L = [seg_to_line(s) for s in corners]  # 4 lines total, 2 per direction
    # We don't know pairing order; try both pairings and keep the one producing a convex quad
    def intersect(l1,l2):
        p = np.cross(l1,l2)
        if abs(p[2]) < 1e-9: return None
        return np.array([p[0]/p[2], p[1]/p[2]], dtype=np.float32)

    # Pairing A:
    ptsA = [intersect(L[0],L[2]), intersect(L[0],L[3]), intersect(L[1],L[3]), intersect(L[1],L[2])]
    # Pairing B:
    ptsB = [intersect(L[0],L[2]), intersect(L[0],L[3]), intersect(L[1],L[2]), intersect(L[1],L[3])]

    def valid(pts):
        if any(p is None for p in pts): return False
        pts = np.stack(pts)
        # area of hull should be > 0
        hull = cv2.convexHull(pts.astype(np.float32))
        return hull is not None and len(hull) >= 4 and cv2.contourArea(hull) > 1e3

    if valid(ptsA):
        return order_points(np.stack(ptsA))
    if valid(ptsB):
        return order_points(np.stack(ptsB))
    return None

# ---- run ----
img_path = r"C:\Users\Owner\Desktop\MIE1517\project\YOLO_dataset\rectangular_tables\images\test_5396.jpg"
img = cv2.imread(img_path)

model = YOLO(WEIGHTS)
raw = get_table_mask_yolo(model, img)
mask = clean_mask(raw)

corners = lsd_table_corners(img, mask)
if corners is None:
    # fallback: minAreaRect (works but less geometrically “correct” under occlusion)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    corners = order_points(cv2.boxPoints(rect))

warped, H = warp_from_corners(img, corners)
cv2.imwrite("table_topdown.png", warped)
