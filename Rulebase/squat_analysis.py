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

def check_squat(landmarks, error_timestamps, current_time):
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

    back_angle = (calculate_angle(left_shoulder, left_hip, left_knee) + calculate_angle(right_shoulder, right_hip, right_knee)) / 2
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    knee_angle = (left_knee_angle + right_knee_angle) / 2
    mid_shoulder = [(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2]
    mid_hip = [(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2]
    tilt_angle = calculate_tilt_angle(mid_hip, mid_shoulder)

    if left_knee_angle > 150 and right_knee_angle > 150:
        if back_angle < 160:
            potential_errors["Back curved while standing"] = f"Back angle: {back_angle:.2f}° < 160°"
    else:
        if back_angle < 90:
            potential_errors["Back too curved"] = f"Back angle: {back_angle:.2f}° < 90°"
        if knee_angle > 120:
            potential_errors["Hips not low enough"] = f"Knee angle: {knee_angle:.2f}° > 120°"
        if tilt_angle > 20:
            potential_errors["Body tilted"] = f"Tilt angle: {tilt_angle:.2f}° > 20°"

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