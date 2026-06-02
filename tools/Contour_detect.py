import cv2
import numpy as np

HEAD_THRESH = 0.75      # Threshold used to separate head from shaft
TIP_IGNORE = 0.15       # Fraction of the profile to ignore at the tapered tip
#FALLBACK_MM_PER_PIXEL = 0.45 # Used if the ArUco marker is not detected


# --- Helper Functions ---
def load_calibration(filepath):
    try:
        with np.load(filepath) as f:
            print("Camera calibration loaded.")
            return f['camera_matrix'], f['dist_coeffs']
    except Exception as e:
        print(f"Warning – calibration not loaded: {e}")
        return None, None

# --- Measurement & Classification ---

def get_standard_screw_size(length_mm, head_dia_mm, shaft_dia_mm, confidence_threshold=0.15):
    """
    Optimized Screw Classifier using a weighted feature matrix and 
    confidence-gated outlier rejection.
    """
    screw_database = {
        6:  {"length": 13.4, "head": 10.0, "shaft": 6.0},
        8:  {"length": 17.8, "head": 13.0, "shaft": 8.0},
        10: {"length": 25.5, "head": 16.0, "shaft": 10.0},
        12: {"length": 31.5, "head": 18.0, "shaft": 12.0},
        16: {"length": 35.7, "head": 24.0, "shaft": 16.0},
        20: {"length": 65.0, "head": 30.0, "shaft": 20.0}
    }
    
    # Priority weighting factors (Must sum to 1.0)
    # Shaft is given massive priority because it defines the fundamental M-size.
    W_SHAFT  = 0.70  
    W_HEAD   = 0.15  
    W_LENGTH = 0.15  

    best_size = "??"  # Default to unknown instead of forcing a wrong match
    min_error = float('inf')
    
    for size, dims in screw_database.items():
        # Compute relative error profiles
        err_length = ((length_mm - dims["length"]) / dims["length"]) ** 2
        err_head   = ((head_dia_mm - dims["head"]) / dims["head"]) ** 2
        err_shaft  = ((shaft_dia_mm - dims["shaft"]) / dims["shaft"]) ** 2
        
        # Apply the feature importance matrix weights
        weighted_error = (W_LENGTH * err_length) + (W_HEAD * err_head) + (W_SHAFT * err_shaft)
        
        if weighted_error < min_error:
            min_error = weighted_error
            best_size = f"M{size}"
            
    # --- OUTLIER REJECTION GATE ---
    # If the closest match has too high of an error, flag it as unrecognizable
    # to protect the end-effector hardware.
    if min_error > confidence_threshold:
        print(f"[CLASSIFIER WARNING] High variance detected (Error: {min_error:.4f}). Flagging object as unknown.")
        return "M??"
        
    return best_size

'''
def get_standard_screw_size(length_mm, head_dia_mm, shaft_dia_mm):
    """
    Snaps the measured physical diameter to the nearest standard metric screw size (M2 to M20).
    """
    standard_sizes = [6, 8, 10, 12, 16, 20]
    closest_size = min(standard_sizes, key=lambda x: abs(x - shaft_dia_mm))
    return closest_size
'''
def find_shaft_in_profile(profile):
    """
    Given a 1-D array of cross-section widths along the screw's long axis
    (already rotated so the screw is horizontal), locate the shaft using T-shape logic.
    """
    SMOOTH_WIN   = 5      # smoothing kernel width (pixels)
    kernel   = np.ones(SMOOTH_WIN) / SMOOTH_WIN
    smoothed = np.convolve(profile, kernel, mode="same")

    max_w  = float(np.max(smoothed))
    length = len(smoothed)

    # Locate the head region
    is_head     = smoothed >= (HEAD_THRESH * max_w)
    head_widths = smoothed[is_head]
    head_dia_px = float(np.median(head_widths)) if len(head_widths) else max_w

    # Shaft zone
    tip_cut  = max(1, int(length * (1.0 - TIP_IGNORE)))
    not_head = ~is_head
    not_head[tip_cut:] = False
    shaft_candidates = smoothed[not_head]

    if len(shaft_candidates) >= 3:
        sorted_c    = np.sort(shaft_candidates)
        bottom_half = sorted_c[: max(1, len(sorted_c) // 2)]
        shaft_dia_px = float(np.median(bottom_half))
    else:
        win = min(5, length)
        best_mean, shaft_dia_px = np.inf, float(np.min(smoothed))
        for i in range(length - win):
            m = float(np.mean(smoothed[i:i + win]))
            if m < best_mean:
                best_mean    = m
                shaft_dia_px = m

    return head_dia_px, shaft_dia_px

def analyze_geometry(polygon, img_shape):
    """
    2. Find the minimum-area bounding rectangle and rotate the mask so the
       screw's long axis is horizontal.
    3. Compute the cross-section profile (width at every pixel column).
    4. Delegate to find_shaft_in_profile() to separate head from shaft.
    """
    # ── Rasterise ──────────────────────────────────────────────────────────
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    # ── Minimum-area bounding rect ─────────────────────────────────────────
    rect = cv2.minAreaRect(polygon)
    center, (w_rect, h_rect), angle = rect
    box = np.int32(cv2.boxPoints(rect))

    # Ensure the long axis maps to the horizontal direction after rotation
    if w_rect < h_rect:
        angle += 90

    # ── Rotate mask (with Canvas Expansion to Prevent Cropping) ────────────
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Grab the dimensions of the original image
    h, w = img_shape[:2]
    
    # Calculate the absolute sine and cosine of the rotation angle
    cos_a = np.abs(M_rot[0, 0])
    sin_a = np.abs(M_rot[0, 1])
    
    # Compute the new bounding dimensions of the image
    new_w = int((h * sin_a) + (w * cos_a))
    new_h = int((h * cos_a) + (w * sin_a))
    
    # Adjust the rotation matrix to take into account the translation
    # This ensures the rotation center maps to the exact center of the new canvas
    M_rot[0, 2] += (new_w / 2) - center[0]
    M_rot[1, 2] += (new_h / 2) - center[1]

    # Warp affine with the newly calculated dimensions
    rotated = cv2.warpAffine(mask, M_rot, (new_w, new_h))

    # ── Cross-section profile ──────────────────────────────────────────────
    col_widths  = np.sum(rotated / 255, axis=0)  # width (px) at each column
    active_cols = np.where(col_widths > 0)[0]

    if len(active_cols) == 0:
        return 0, 0, 0, box, M_rot, center, angle, np.array([]), rotated

    profile    = col_widths[active_cols]   # strip leading/trailing zeros
    length_px  = len(profile)

    # ── Shaft detection ────────────────────────────────────────────────────
    head_dia_px, shaft_dia_px = find_shaft_in_profile(profile)

    # Column indices (in rotated space) where the shaft was detected.
    smoothed     = np.convolve(profile, np.ones(5) / 5, mode="same")
    is_shaft_col = (smoothed < HEAD_THRESH * np.max(smoothed))
    shaft_cols   = active_cols[is_shaft_col]

    return length_px, head_dia_px, shaft_dia_px, box, M_rot, center, angle, shaft_cols, rotated
    
def extract_screw_polygon(cropped_img):
    """
    Robust contour extraction optimized for shadows, poor gamma, and reflections.
    """
    # Convert to Grayscale
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    # Fix Gamma / Contrast locally using CLAHE
    # clipLimit controls the contrast limit. 2.0 is usually a good starting point.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Edge-preserving blur (Bilateral Filter)
    # Reduces noise and smooths shadow gradients without blurring the hard edge of the screw
    blurred = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Thresholding
    # Note: If your background is very dark and the screw is bright, remove cv2.THRESH_BINARY_INV
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological Cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # 'Opening' removes small false-positive blobs (like detached shadows or dirt)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 'Closing' fills in small holes inside the screw mask (like bright metallic reflections)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # 7. Assume the largest contour is the screw
    largest_contour = max(contours, key=cv2.contourArea)
    
    return largest_contour
