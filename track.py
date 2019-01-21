import cv2
import numpy as np

# capture the video
cap = cv2.VideoCapture('vid/capture.mp4')

# Create old frame
_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Lucas kanade params
lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# Mouse function
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points, p1, p2
    if event == cv2.EVENT_LBUTTONDOWN:
        # get the point of interest
        point = (x, y)
        # save the point and print it to console
        p1, p2 = x, y
        print('{} {}'.format(x, y))
        # necessary variables
        point_selected = True
        # construct the old point for LK method
        old_points = np.array([[x, y]], dtype=np.float32)


# create output window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)
# initial variables
point_selected = False
point = ()
old_points = np.array([[]])
# start looping through the video
while True:
    # read recent frame
    _, frame = cap.read()
    # convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if point_selected is True:
        # cv2.circle(frame, point, 20, (0, 0, 255), 2)
        # calculate optical flow pyramid
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        # move the old frame to new frame and do the same for the point
        old_gray = gray_frame.copy()
        old_points = new_points
        # get a 1D view of the point
        x, y = new_points.ravel()
        # print the new points in the new frame
        print('{} {}'.format(x, y))
        # encircle the point of interest
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        # detecting direction of motion
        if x < p1:
            print('left')
            # print direction on the frame
            cv2.putText(frame, 'LEFT  <-', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
        elif x > p1:
            print('right')
            # print direction on the frame
            cv2.putText(frame, 'RIGHT  ->', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
        # update the points
        p1, p2 = x, y
    # show the frame
    cv2.imshow("Frame", frame)
    # wait for keypress
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
# release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()