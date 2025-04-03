import numpy as np

def calculate_angle(p1, p2, p3):
    a = np.array([p1.x, p1.y]) - np.array([p2.x, p2.y])
    b = np.array([p3.x, p3.y]) - np.array([p2.x, p2.y])
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def calculate_tilt_angle(p1, p2):
    vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    y_axis = np.array([0, -1])
    cosine_angle = np.dot(vector, y_axis) / (np.linalg.norm(vector) * np.linalg.norm(y_axis))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    if p2[1] > p1[1]:
        return 180 - angle
    return angle

def check_lunges(landmarks, error_timestamps, current_time):
    potential_errors = {}
    confirmed_errors = []

    left_shoulder = landmarks[11]
    left_hip = landmarks[23]
    left_knee = landmarks[25]
    left_ankle = landmarks[27]
    right_shoulder = landmarks[12]
    right_hip = landmarks[24]
    right_knee = landmarks[26]
    right_ankle = landmarks[28]

    if left_knee.y < right_knee.y:
        front_knee, front_hip, front_ankle = left_knee, left_hip, left_ankle
        back_knee, back_hip, back_shoulder, back_ankle = right_knee, right_hip, right_shoulder, right_ankle
    else:
        front_knee, front_hip, front_ankle = right_knee, right_hip, right_ankle
        back_knee, back_hip, back_shoulder, back_ankle = left_knee, left_hip, left_shoulder, left_ankle

    back_angle = calculate_angle(back_shoulder, back_hip, back_knee)
    front_knee_angle = calculate_angle(front_hip, front_knee, front_ankle)
    back_knee_angle = calculate_angle(back_hip, back_knee, back_ankle)
    mid_shoulder = np.array([(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2])
    mid_hip = np.array([(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2])
    tilt_angle = calculate_tilt_angle(mid_hip, mid_shoulder)

    is_standing = front_knee_angle > 150 and back_knee_angle > 150
    if is_standing:
        if back_angle < 160:
            potential_errors["Back too curved"] = f"Back angle: {back_angle:.2f}° < 160°"
        if tilt_angle > 20:
            potential_errors["Body tilted"] = f"Tilt angle: {tilt_angle:.2f}° > 20°"
    else:
        if tilt_angle > 20:
            potential_errors["Body tilted"] = f"Tilt angle: {tilt_angle:.2f}° > 20°"
        if mid_shoulder[1] > mid_hip[1] + 0.05:
            potential_errors["Not in lunge position"] = f"Shoulder Y: {mid_shoulder[1]:.2f} > Hip Y: {mid_hip[1]:.2f}"
        if back_angle < 160:
            potential_errors["Back too curved"] = f"Back angle: {back_angle:.2f}° < 160°"
        if front_knee_angle < 70 or front_knee_angle > 110:
            potential_errors["Front knee incorrect"] = f"Front knee angle: {front_knee_angle:.2f}° not ~90°"
        if back_knee.y > 0.9:
            potential_errors["Back knee too high"] = f"Back knee Y: {back_knee.y:.2f} > 0.9"
        if back_knee_angle > 120:
            potential_errors["Back knee not bent enough"] = f"Back knee angle: {back_knee_angle:.2f}° > 120°"

    for error, reason in potential_errors.items():
        if error not in error_timestamps:
            error_timestamps[error] = current_time
        elif current_time - error_timestamps[error] > 0.5:
            confirmed_errors.append(error)

    for error in list(error_timestamps.keys()):
        if error not in potential_errors:
            del error_timestamps[error]

    status = "Correct" if not confirmed_errors else "Incorrect"
    return status, confirmed_errors