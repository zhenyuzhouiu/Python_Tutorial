import os
import cv2

video_path = r'C:\Users\ZhenyuZHOU\Pictures\Finger_Knuckle'
dst_path = r'C:\Users\ZhenyuZHOU\Pictures\Segment_Finger_Knuckle'
sub_name = os.listdir(video_path)
sub_name = ['007']

for s in sub_name:
    sub_path = os.path.join(video_path, s)
    dst_sub_path = os.path.join(dst_path, s)
    if not os.path.exists(dst_sub_path):
        os.mkdir(dst_sub_path)
    sub_sub_name = os.listdir(sub_path)
    sub_sub_path = os.path.join(sub_path, sub_sub_name[0])
    dst_sub_sub_path = os.path.join(dst_sub_path, sub_sub_name[0])
    if not os.path.exists(dst_sub_sub_path):
        os.mkdir(dst_sub_sub_path)
    video_name = os.listdir(sub_sub_path)
    for v in video_name:
        video_file = os.path.join(sub_sub_path, v)
        dst_video_path = os.path.join(dst_sub_sub_path, v.split('.')[0])
        if not os.path.exists(dst_video_path):
            os.mkdir(dst_video_path)

        # read video file
        videoCapture = cv2.VideoCapture(video_file)
        # read camera
        # videoCapture = cv2.VideoCapture(1)

        # read frame
        success, frame = videoCapture.read()
        i = 0
        timeF = 30
        j = 0
        while success:
            if (i % timeF == 0):
                cv2.imwrite(dst_video_path + "/frame%d.jpg" % j, frame)
                j = j + 1
            success, frame = videoCapture.read()
            print('Read a new frame: ', success)
            i += 1
