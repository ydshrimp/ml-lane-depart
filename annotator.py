import cv2
import cv
import os
import time

class Annotator:
    def __init__(self, video_src):
        self.video_src = video_src
        self.cap = cv2.VideoCapture(video_src)
        self.saveFile = None

    def record(self, framenum, label):
        timestamp = self.cap.get(cv.CV_CAP_PROP_POS_MSEC)
        self.saveFile.write(str(framenum) + ',' + str(timestamp) + ',' + label + '\n')

    def play_video(self, pathName, fileName):
        pathName, fileNameExt = os.path.split(self.video_src)

        if not os.path.exists(pathName + '/train_labels/'):
            os.makedirs(pathName + '/train_labels')
        print 'Writing to: '+ pathName + '/train_labels/' + str(fileName) + '_labels.csv'
        if os.path.exists(pathName + '/train_labels/' + str(fileName) + '_labels.csv'):
            return
        self.saveFile = open(pathName + '/train_labels/' + str(fileName) + '_labels.csv','w')
        self.saveFile.write('FrameIdx, TimeStamp, Action' + '\n')
        

        videoScreen = cv2.namedWindow('Video Player', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(videoScreen, 600, 400)
        frameidx = 0
        ffIdx = 1 # Fast forward speed (in num frames skipped)
        while(self.cap.isOpened()):
            isdone = False
            for l in range(0, ffIdx):
                frameidx += 1
                ret = self.cap.grab()
                if ret == False:
                    isdone = True
                    break
            ret, frame = self.cap.retrieve()
            
            if ret==True:
                #frame = cv2.flip(frame,1)  #optional flip in case original video is flipped
                resized_image = cv2.resize(frame, (600, 400))
                
                cv2.imshow(videoScreen, resized_image)
                ch = cv2.waitKey(1) & 0xFF

                if ch == ord('f'):
                    self.record(frameidx, 'ignore')
                    print "Ignore"

                if ch == ord('w'):
                    self.record(frameidx, 'end action')
                    print "End of action"
                if ch == ord('s'):
                    self.record(frameidx, 'stop')
                    print "Stop"
                if ch == ord('a'):
                    self.record(frameidx, 'lturn')
                    print "Left TURN"
                if ch == ord('d'):
                    self.record(frameidx, 'rturn')
                    print "Right TURN"

                if ch == ord('q'):
                    self.record(frameidx, 'lchange')
                    print "Left LANE CHANGE"
                if ch == ord('e'):
                    self.record(frameidx, 'rchange')
                    print "Right LANE CHANGE"

                if ch == ord('p'):
                    ffIdx += 1
                    print "Speed: " + str(ffIdx)
                if ch == ord('o'):
                    if (ffIdx > 0):
                        ffIdx -= 1
                        print "Speed: " + str(ffIdx)
                if ch == 27:
                    break
                
                if isdone == True:
                    self.record(frameidx, 'eof')
                    frameidx = 0
                    print "Reached end of video. Replaying the video. Press ESC if you want to end.\n"
            else:
                break
                self.record(frameidx, 'eof')
                frameidx = 0
                print "Reached end of video. Replaying the video. Press ESC if you want to end.\n"

        useful = raw_input('Was this useful? y or n? \n')
        if useful == 'n':
            self.record(frameidx, 'delete')
            
        self.cap.release()
        self.saveFile.close()
        cv2.destroyAllWindows()


def main():
    import sys
    import tkFileDialog
    video_src_directory = tkFileDialog.askdirectory()

    for root, dirs, filenames in os.walk(video_src_directory):
        for f in filenames:
            fileName, fileExtension = os.path.splitext(f)
            if (fileExtension == '.mov'):
                usrkey = raw_input ('Press any keys to continue on to the next file: ' + f + '\n')
                print usrkey
                print 'Running: ' + os.path.join(root, f) + '\n'
                Annotator(os.path.join(root, f)).play_video(os.path.join(root, f), fileName)
    print 'We are all done! Program is exiting...\n'

if __name__ == '__main__':
    main()
