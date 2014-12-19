import numpy as np
import cv2
import time
# TO DO
# Possible improvement:
# Remove points once they've exited masked area (so we don't take into account points that move left and right along the line of perspective)
# Use HoughLinesP and track endpoints of detected lines (may end up tracking road markings?)

start_time = time.time();
cap = cv2.VideoCapture('.//test_videos//lanechange2.mov')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 5,
                       blockSize = 7)
threshold = 10 # hardcoded for now

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Generate mask for the center of image
fMask = np.zeros_like(old_frame[:,:,1])
fMask[(old_frame.shape[0]/9):(7*old_frame.shape[0]/9), 2*old_frame.shape[1]/6:(4*old_frame.shape[1]/6)] = 255

p0 = cv2.goodFeaturesToTrack(old_gray, mask = fMask, **feature_params)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
cv2.cornerSubPix(old_gray, p0, (5,5), (-1,-1), criteria)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# Initialize logging
logfile = open('opticalflow_log.csv', 'w')
logfile.write('TimeStamp,NumPoints,XDisplacement'+ '\n')
logfile_meandisp = open('opticalflow_meandisp_log.csv', 'w')
logfile_meandisp.write('TimeStamp,MeanXDisplacement'+ '\n')

frame_idx = 0
while(1):
    frame_idx += 1

    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points (no error and within ROI)
    good_new_array = []
    good_old_array= []
    for i, g1 in enumerate(p1[st==1]):
        if (fMask[int(g1[1])][int(g1[0])] > 0):
            good_new_array.append([g1[0], g1[1]])
            good_old_array.append(p0[st==1][i])
    good_new = np.array(good_new_array)
    good_old = np.array(good_old_array)

    # print 'Num pts: ' + str(len(good_new))

    # Draw corners
    xdisplacement = []
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.circle(frame, (a,b),5,color[i].tolist(),-5)

        # Log x displacement for each corner
        xdisplacement.append(new[0]-old[0])
        logfile.write("%0.4f" % (time.time() - start_time) + ',' + str(len(good_new)) + ',' + str(xdisplacement[-1]) + '\n')


    # Find mean displacement
    mean_xdisp = sum(xdisplacement)/float(len(xdisplacement))
    logfile_meandisp.write("%0.4f" % (time.time() - start_time) + ',' + str(mean_xdisp) + '\n')
    print mean_xdisp

    # Show frames
    fMaskColor = cv2.cvtColor(fMask, cv2.COLOR_GRAY2BGR)
    frame_overlay = cv2.addWeighted(frame, 0.8, fMaskColor, 0.2, 0)
    cv2.imshow('frame', frame_overlay)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

    # recalculate corners if number of points < threshold
    if (len(good_new) <= threshold):
        print "recalculating corners\n"
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask = fMask, **feature_params)
        cv2.cornerSubPix(frame_gray, p0, (5,5), (-1,-1), criteria)
cv2.destroyAllWindows()
cap.release()

