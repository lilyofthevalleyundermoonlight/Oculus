import cv2 
import numpy as np
import dlib
from math import hypot
import time
import random

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_SIMPLEX

origin_time = time.time()
start_time_blinking = time.time()
start_time_exercises = time.time()
start_time_break = time.time()
time_interval_blinking = 60
time_interval_exercises = 1200
time_interval_break = 3600
break_time = 300
break_started = False


message = ""

instructions = [1, 2, 3, 4]
direction_movements = []

blink_count = 0
eye_closed = False

look_up = False
look_down = False
look_right = False
look_left = False

def midpoint(p1, p2):
  return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0, 2))
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0, 2))

    hor_line_length = hypot(left_point[0] - right_point[0], left_point[1] - right_point[1])
    ver_line_length = hypot(center_top[0] - center_bottom[0], center_top[1] - center_bottom[1])

    ratio = hor_line_length/ ver_line_length
    return ratio

while True:
  ret, frame = cap.read()
  if ret is False:
    break
  width = frame.shape[1]
  estimated_time = time.time()
  estimated_time_blinking = time.time() - start_time_blinking
  estimated_time_exercises = time.time() - start_time_exercises
  estimated_time_break = time.time() - start_time_break
  hours, remainder = divmod(estimated_time - origin_time, 3600)
  minutes, seconds = divmod(remainder, 60)
  screen_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
  cv2.putText(frame, "Screen time: " + screen_time, (width//2 - 100, 50), font, 1, (0, 255, 0), 3)
  
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray = gray.astype('uint8')
  gray = cv2.equalizeHist(gray)
  
  faces = detector(gray)

  cv2.putText(frame, message, (10, 200), font, 1, (0, 255, 0), 2)

  if estimated_time_break >= time_interval_break:
    if not faces:  # No face detected
        if not break_started:
            break_start_time = time.time()
            break_started = True
        elapsed_break_time = time.time() - break_start_time
        if elapsed_break_time >= break_time:
            message = "Break complete. You can return."
            break_started = False
            start_time_break = time.time()  # Reset break timer
            start_time_exercises = time.time()
            start_time_blinking = time.time()
        else:
            remaining_time = break_time - elapsed_break_time
            message = f"Break in progress. Time left: {int(remaining_time)} s."
    else:  # Face detected
        if break_started and elapsed_break_time < break_time:
            message = f"Break interrupted. You spent {int(elapsed_break_time)} s on break."
            break_started = False
            start_time_break = time.time()  # Reset break timer
            start_time_exercises = time.time()
            start_time_blinking = time.time()
        else:
            message = "Break not taken. Please rest."

  for face in faces:
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()

    landmarks = predictor(gray, face)

    left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                       (landmarks.part(37).x, landmarks.part(37).y),
                       (landmarks.part(38).x, landmarks.part(38).y),
                       (landmarks.part(39).x, landmarks.part(39).y),
                       (landmarks.part(40).x, landmarks.part(40).y),
                       (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

    right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                          (landmarks.part(43).x, landmarks.part(43).y),
                          (landmarks.part(44).x, landmarks.part(44).y),
                          (landmarks.part(45).x, landmarks.part(45).y),
                          (landmarks.part(46).x, landmarks.part(46).y),
                          (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

    left_eye_min_x = np.min(left_eye_region[:, 0])
    left_eye_max_x = np.max(left_eye_region[:, 0])
    left_eye_min_y = np.min(left_eye_region[:, 1])
    left_eye_max_y = np.max(left_eye_region[:, 1])

    left_eye = gray[left_eye_min_y: left_eye_max_y, left_eye_min_x: left_eye_max_x]
    left_rows, left_cols = left_eye.shape
    gray_left_eye = cv2.GaussianBlur(left_eye, (3, 5), 0)
    
    left_avg_brightness = np.mean(gray_left_eye)
    left_dynamic_threshold = np.clip(left_avg_brightness * 0.8, 30, 200)
    _, threshold_left_eye = cv2.threshold(gray_left_eye, left_dynamic_threshold, 255, cv2.THRESH_BINARY_INV)

    left_contours, _ = cv2.findContours(threshold_left_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    left_contours = sorted(left_contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for left_cnt in left_contours:
        (x, y, w, h) = cv2.boundingRect(left_cnt)
        left_rect_x1 = left_eye_min_x + x
        left_rect_y1 = left_eye_min_y + y
        left_rect_x2 = left_eye_min_x + x + w
        left_rect_y2 = left_eye_min_y + y + h
        cv2.rectangle(frame, (left_rect_x1, left_rect_y1), (left_rect_x2, left_rect_y2), (255, 0, 0), 2)

        left_rect_x = (left_rect_x1 + left_rect_x2) / 2 
        left_rect_y = (left_rect_y1 + left_rect_y2) / 2
        break

    
    right_eye_min_x = np.min(right_eye_region[:, 0])
    right_eye_max_x = np.max(right_eye_region[:, 0])
    right_eye_min_y = np.min(right_eye_region[:, 1])
    right_eye_max_y = np.max(right_eye_region[:, 1])

    right_eye = gray[right_eye_min_y: right_eye_max_y, right_eye_min_x: right_eye_max_x]
    gray_right_eye = cv2.GaussianBlur(right_eye, (3, 3), 0)

    right_avg_brightness = np.mean(gray_right_eye)
    right_dynamic_threshold = np.clip(right_avg_brightness * 0.8, 30, 200)
    _, threshold_right_eye = cv2.threshold(gray_right_eye, right_dynamic_threshold, 255, cv2.THRESH_BINARY_INV)

    right_contours, _ = cv2.findContours(threshold_right_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    right_contours = sorted(right_contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for right_cnt in right_contours:
        (x, y, w, h) = cv2.boundingRect(right_cnt)
        right_rect_x1 = right_eye_min_x + x
        right_rect_y1 = right_eye_min_y + y
        right_rect_x2 = right_eye_min_x + x + w
        right_rect_y2 = right_eye_min_y + y + h
        cv2.rectangle(frame, (right_rect_x1, right_rect_y1), (right_rect_x2, right_rect_y2), (255, 0, 0), 2)

        right_rect_x = (right_rect_x1 + right_rect_x2) / 2 
        right_rect_y = (right_rect_y1 + right_rect_y2) / 2 
        break
    
    left_left_point = (landmarks.part(36).x, landmarks.part(36).y)
    left_right_point = (landmarks.part(39).x, landmarks.part(39).y)
    left_center_top = midpoint(landmarks.part(37), landmarks.part(38))
    left_center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

    left_hor_line = cv2.line(frame, left_left_point, left_right_point, (0, 255, 0, 2))
    left_ver_line = cv2.line(frame, left_center_top, left_center_bottom, (0, 255, 0, 2))

    left_x_intersect = left_center_top[0]
    left_y_intersect = left_left_point[1]
    
    left_hor_line_length = hypot(left_left_point[0] - left_right_point[0], left_left_point[1] - left_right_point[1])
    left_ver_line_length = hypot(left_center_top[0] - left_center_bottom[0], left_center_top[1] - left_center_bottom[1])

    left_eye_ratio = left_hor_line_length/ left_ver_line_length

    right_left_point = (landmarks.part(42).x, landmarks.part(42).y)
    right_right_point = (landmarks.part(45).x, landmarks.part(45).y)
    right_center_top = midpoint(landmarks.part(43), landmarks.part(44))
    right_center_bottom = midpoint(landmarks.part(47), landmarks.part(46))

    right_hor_line = cv2.line(frame, right_left_point, right_right_point, (0, 255, 0, 2))
    right_ver_line = cv2.line(frame, right_center_top, right_center_bottom, (0, 255, 0, 2))

    right_x_intersect = right_center_top[0]
    right_y_intersect = right_left_point[1]
    right_hor_line_length = hypot(right_left_point[0] - right_right_point[0], right_left_point[1] - right_right_point[1])
    right_ver_line_length = hypot(right_center_top[0] - right_center_bottom[0], right_center_top[1] - right_center_bottom[1])

    right_eye_ratio = right_hor_line_length/ right_ver_line_length   
    
    
    blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

    if blinking_ratio > 5:
      if not eye_closed:
                blink_count += 1
                eye_closed = True
                blink_duration = 0
      cv2.putText(frame, "BLINKING", (10, 50), font, 1, (255, 0, 0), 3)
      
    else:
       if eye_closed:
          blink_duration += 1  # Count the number of frames the eyes are closed
          eye_closed = False
    
    if estimated_time_exercises >= time_interval_exercises and estimated_time_break < time_interval_break:
        if len(set(direction_movements)) < 4:  # Check if all movements (1=Right, 2=Left, 3=Up, 4=Down) are done

            look_left, look_right, look_down, look_up = False, False, False, False
            if 1 not in direction_movements:
                message = "Look Right"
                if left_rect_x < left_x_intersect - 0.2 * left_hor_line_length or right_rect_x < right_x_intersect - 0.2 * right_hor_line_length:
                    cv2.putText(frame, "RIGHT", (10, 100), font, 1, (0, 0, 255), 3)
                    look_right = True
                    direction_movements.append(1)
            elif 2 not in direction_movements:
                message = "Look Left"
                if left_rect_x > left_x_intersect + 0.2 * left_hor_line_length or right_rect_x > right_x_intersect + 0.2 * right_hor_line_length:
                    cv2.putText(frame, "LEFT", (10, 100), font, 1, (0, 0, 255), 3)
                    look_left = True
                    direction_movements.append(2)
            elif 3 not in direction_movements:
                message = "Look Up"
                if left_rect_y < left_y_intersect - 0.3 * left_ver_line_length or right_rect_y < right_y_intersect - 0.75 * right_ver_line_length:
                    cv2.putText(frame, "UP", (10, 100), font, 1, (0, 0, 255), 3)
                    look_up = True
                    direction_movements.append(3)
            elif 4 not in direction_movements:
                message = "Look Down"
                if left_rect_y > left_y_intersect or right_rect_y > right_y_intersect:
                    cv2.putText(frame, "DOWN", (10, 100), font, 1, (0, 0, 255), 3)
                    look_down = True
                    direction_movements.append(4)
        else:
        # Exercise complete
            message = "Good job! Exercise complete."
            start_time_exercises = time.time()  # Reset exercise timer
            start_time_blinking = time.time()
            direction_movements.clear()  # Clear the movement tracker for the next round
    
    elif estimated_time_blinking >= time_interval_blinking and estimated_time_exercises < time_interval_exercises and estimated_time_break < time_interval_break:
        if blink_count < 15:
            message = f"Blinks/min: {blink_count}. Blink {15 - blink_count} more times."
        else:
            message ="Blinking is healthy. Keep it up!"
            start_time_blinking = time.time()
            blink_count = 0 
   
  cv2.imshow("Frame", frame)

  key = cv2.waitKey(1)
  if key == 27:
    break

cap.release()
cv2.destroyAllWindows()