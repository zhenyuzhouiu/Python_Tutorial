import os
import shutil

source_path = '/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/3Dfingerknuckle/3D Finger Knuckle Database New (20190711)/segmented_data_down'
dest_path = '/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/3Dfingerknuckle/3D Finger Knuckle Database New (20190711)/two-session'

sub_names = os.listdir(source_path)

for i in sub_names:
    sub_path = os.path.join(source_path, i)
    session_name = os.listdir(sub_path)
    for j in session_name:
        session_path = os.path.join(sub_path, j)
        knuckle = os.listdir(session_path)
        for k in knuckle:
            knuckle_path = os.path.join(session_path, k)
            sample_name = os.listdir(knuckle_path)
            n_sample = 0
            for l in sample_name:
                sample_path = os.path.join(knuckle_path, l)
                sample2d_path = os.path.join(sample_path, '2D')
                img_path = os.path.join(sample2d_path, 'stack.bmp')
                dest_knuckle = os.path.join(dest_path, k)
                if not os.path.exists(dest_knuckle):
                    os.mkdir(dest_knuckle)
                dest_session = os.path.join(dest_knuckle, j)
                if not os.path.exists(dest_session):
                    os.mkdir(dest_session)
                dest_subject = os.path.join(dest_session, i)
                if not os.path.exists(dest_subject):
                    os.mkdir(dest_subject)
                dest_img = os.path.join(dest_subject, 'stack{}.bmp'.format(n_sample))
                n_sample += 1
                shutil.copy(img_path, dest_img)


