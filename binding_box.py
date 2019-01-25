import cv2
import numpy as np

cap = cv2.VideoCapture("video/capture.mp4")
_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# from first frame
p1 = (267 + 402)//2
p2 = (136 + 247)//2
center = (p1, p2)

old_points = np.array([[p1, p2]], dtype=np.float32)

#cv2.circle(frame, center, 1, (0, 0, 255), 1)
while True:
    # read recent frame
    _, frame = cap.read()
    # convert to grayscale
    # cv2.rectangle(frame, (267, 136), (402, 247), (0, 0, 255), 2)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
    old_gray = gray_frame.copy()
    old_points = new_points
    x, y = new_points.ravel()

    #cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    point_x1 = int(x + 135//2)
    #print(point_x1)
    point_y1 = int(y + 111//2)
    #print(point_y1)
    point_x2 = int(x - 135//2)
    #print(point_x2)
    point_y2 = int(y - 111//2)
    #print(point_y2)
    cv2.rectangle(frame, (point_x1, point_y1), (point_x2, point_y2), (0, 0, 255), 2)

    if x < p1:
        print('left')
        cv2.putText(frame, 'LEFT', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
    elif x > p1:
        print('right')
        cv2.putText(frame, 'RIGHT', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
    p1, p2 = x, y
    cv2.imshow("Frame", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
# release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()