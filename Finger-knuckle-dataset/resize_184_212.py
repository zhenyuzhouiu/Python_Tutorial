import os
import cv2

src_path = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV3/GUI_Segment/Session_2/"
dst_path = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV3/GUI_Segment/Keep_ratio/Session_2/"

subject = os.listdir(src_path)

ratio = 26 / 23
resize_h = 208
resize_w = 184

for s in subject:
    subject_path = os.path.join(src_path, s)
    dst_subject_path = os.path.join(dst_path, s)
    if not os.path.exists(dst_subject_path):
        os.mkdir(dst_subject_path)
    images = os.listdir(subject_path)
    for i in images:
        image_path = os.path.join(subject_path, i)
        dst_image_path = os.path.join(dst_subject_path, i)
        src_img = cv2.imread(image_path)
        h, w, c = src_img.shape
        dest_w = h / ratio
        dest_h = w * ratio
        if dest_w > w:
            crop_h = int((h - dest_h) / 2)
            if crop_h == 0:
                crop_h = 1
            crop_image = src_img[crop_h-1:crop_h+int(dest_h), :, :]
            tmp_h, tmp_w, tmp_c = crop_image.shape
            print("ratio: " + str(tmp_h/tmp_w))
        elif dest_h > h:
            crop_w = int((w - dest_w) / 2)
            if crop_w == 0:
                crop_w = 1
            crop_image = src_img[:, crop_w-1:crop_w+int(dest_w), :]
            tmp_h, tmp_w, tmp_c = crop_image.shape
            print("ratio: " + str(tmp_h/tmp_w))
        else:
            crop_image = src_img
            tmp_h, tmp_w, tmp_c = crop_image.shape
            print("ratio: " + str(tmp_h/tmp_w))
            print("don't need change ratio")

        resize_image = cv2.resize(crop_image, (resize_w, resize_h))
        # cv2.imshow('demo show', resize_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(dst_image_path, resize_image)

