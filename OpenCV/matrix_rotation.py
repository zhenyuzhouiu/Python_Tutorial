import cv2
import numpy as np

def rotate(vector, theta, rotation_around=None) -> np.ndarray:
    """
    reference: https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
    :param vector: list of length 2 OR
                   list of list where inner list has size 2 OR
                   1D numpy array of length 2 OR
                   2D numpy array of size (number of points, 2)
    :param theta: rotation angle in degree (+ve value of anti-clockwise rotation)
    :param rotation_around: "vector" will be rotated around this point,
                    otherwise [0, 0] will be considered as rotation axis
    :return: rotated "vector" about "theta" degree around rotation
             axis "rotation_around" numpy array
    """
    vector = np.array(vector)

    if vector.ndim == 1:
        vector = vector[np.newaxis, :]

    if rotation_around is not None:
        vector = vector - rotation_around

    vector = vector.T

    theta = np.radians(theta)

    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    output: np.ndarray = (rotation_matrix @ vector).T

    if rotation_around is not None:
        output = output + rotation_around

    return output.squeeze()


if __name__ == '__main__':
    angle = 30
    print(rotate([1, 0], 30))  # passing one point
    print(rotate([[1, 0], [0, 1]], 30))  # passing multiple points

    image_path = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV1/Segmented/GUI_Seg/major/dataset/train_set/1/1_1-0.jpg"
    image_path1 = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV1/Segmented/GUI_Seg/major/dataset/train_set/2/2_3-0.jpg"
    image_path2 = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV1/Segmented/GUI_Seg/major/dataset/train_set/3/3_1-0.jpg"

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # image = np.expand_dims(image, axis=2)
    h, w = image.shape

    rotated_image = rotate(image, theta=5, rotation_around=[h/2, w/2])

    cv2.imshow('src', image)
    cv2.imshow('rotated', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

