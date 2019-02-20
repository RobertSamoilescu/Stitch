import numpy as np
import cv2

def get_translation(dx, dy):
    # translation matrix
    T = np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])

    return T

def get_y_rotation(angle):
    R = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])

    return R

def get_z_rotation(angle):
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])

    return R

def scale(image, scale):
    height, width, _ = image.shape
    img = cv2.resize(image, None, fx=scale, fy=scale)
    new_height, new_width, _ = img.shape

    if scale > 1.0:
        width_offset = (new_width - width) // 2
        height_offset = (new_height - height)
        img = img[height_offset:height_offset + height, width_offset:width_offset + width, :]
        return img

    final_img = np.zeros(image.shape, dtype=np.uint8)
    width_offset = (width - new_width) // 2
    height_offset = (height - new_height) //2

    final_img[height_offset:height_offset + new_height, width_offset:width_offset + new_width, :] = img
    return final_img


if __name__ == "__main__":
    STEP = 1e-3

    # camera params
    K = np.array([
        [1173.122620, 0.000000, 969.335924],
        [0.000000, 1179.612539, 549.524382],
        [0.000000, 0.000000, 1.000000]
    ])

    leftCap = cv2.VideoCapture("/home/robert/PycharmProjects/Final_Simulator/test_data/0ba94a1ed2e0449c-1.mov")
    centerCap = cv2.VideoCapture("/home/robert/PycharmProjects/Final_Simulator/test_data/0ba94a1ed2e0449c-0.mov")
    rightCap = cv2.VideoCapture("/home/robert/PycharmProjects/Final_Simulator/test_data/0ba94a1ed2e0449c-2.mov")

    retA, imageA = leftCap.read()
    retB, imageB = centerCap.read()
    retC, imageC = rightCap.read()

    shape = imageA.shape
    imgA = np.zeros((shape[0], 3 * shape[1], 3), dtype=np.uint8)
    imgA[:, 0:shape[1]] = imageA

    imgB = np.zeros((shape[0], 3 * shape[1], 3), dtype=np.uint8)
    imgB[:, shape[1]:2 * shape[1]] = imageB

    imgC = np.zeros((shape[0], 3 * shape[1], 3), dtype=np.uint8)
    imgC[:, 2 * shape[1]:] = imageC

    S = np.array([
            [360./1080., 0., 0.],
            [0., 640./1920., 0.],
            [0., 0., 1.]
    ])
    K = np.matmul(S, K)
    K[0, 2] += imageA.shape[1]

    camera = "left"
    dx_left = 0.0; dx_right = 0.0
    dy_left = 0.0; dy_right = 0.0
    dz_left = 1.0; dz_right = 1.0
    alpha_left = 0.0; alpha_right = 0.0
    beta_left = 0.0; beta_rigth = 0.0

    while True:
        print("dx_left {} dy_left {} dz_left {}".format(dx_left, dy_left, dz_left))
        print("alpha_left {}".format(alpha_left))
        print("beta_left {}".format(beta_left))
        print("=" * 10)

        print("dx_right {} dy_right {} dz_right {}".format(dx_right, dy_right, dz_right))
        print("alpha_right {}".format(alpha_right))
        print("beta_right {}".format(beta_rigth))
        print("=" * 10)

        c = cv2.waitKey(0)
        if c & 0xFF == ord('a'):
            if camera == "left":
                dx_left -= STEP
            else:
                dx_right -= STEP
        elif c & 0xFF == ord('d'):
            if camera == "left":
                dx_left += STEP
            else:
                dx_right += STEP
        elif c & 0xFF == ord('w'):
            if camera == "left":
                dy_left -= STEP
            else:
                dy_right -= STEP
        elif c & 0xFF == ord('s'):
            if camera == "left":
                dy_left += STEP
            else:
                dy_right += STEP
        elif c & 0xFF == ord('z'):
            if camera == "left":
                dz_left -= STEP
            else:
                dz_right -= STEP
        elif c & 0xFF == ord('x'):
            if camera == "left":
                dz_left += STEP
            else:
                dz_right += STEP
        elif c & 0xFF == ord('q'):
            if camera == "left":
                alpha_left -= STEP
            else:
                alpha_right -= STEP
        elif c & 0xFF == ord('e'):
            if camera == "left":
                alpha_left += STEP
            else:
                alpha_right += STEP
        elif c & 0xFF == ord('r'):
            if camera == "left":
                beta_left -= STEP
            else:
                beta_left += STEP
        elif c & 0xFF == ord('t'):
            if camera == "left":
                beta_left += STEP
            else:
                beta_rigth += STEP
        elif c & 0xFF == 27: # esc
            break
        elif c & 0xFF == 32: # space
            camera = "right" if camera == "left" else "left"
            print("Camera switched")

        A = np.matmul(get_translation(dx_left, dy_left), np.matmul(get_y_rotation(alpha_left), get_z_rotation(beta_left)))
        H = np.matmul(np.matmul(K, A), np.linalg.inv(K))
        left = cv2.warpPerspective(imgA, H, (imgA.shape[1], imgA.shape[0]), flags=cv2.INTER_LINEAR)
        left = scale(left, dz_left)

        A = np.matmul(get_translation(dx_right, dy_right), np.matmul(get_y_rotation(alpha_right), get_z_rotation(beta_rigth)))
        H = np.matmul(np.matmul(K, A), np.linalg.inv(K))
        right = cv2.warpPerspective(imgC, H, (imgC.shape[1], imgC.shape[0]), flags=cv2.INTER_LINEAR)
        right = scale(right, dz_right)

        result = np.zeros_like(imgB)
        # result[:, :shape[1]] += left[:, :shape[1]] // 2
        # result[:, 2*shape[1]:] += right[:, 2*shape[1]:] // 2

        result += left // 2
        result += right // 2
        result[:, shape[1]:2*shape[1]] += imageB // 2
        cv2.imshow("Calibration", result)

    A = np.matmul(get_translation(dx_left, dy_left), np.matmul(get_y_rotation(alpha_left), get_z_rotation(beta_left)))
    H_left = np.matmul(np.matmul(K, A), np.linalg.inv(K))

    A = np.matmul(get_translation(dx_right, dy_right), np.matmul(get_y_rotation(alpha_right), get_z_rotation(beta_rigth)))
    H_right = np.matmul(np.matmul(K, A), np.linalg.inv(K))

    # save matrices
    import pickle
    with open("matrix.pkl", "wb") as output:
        d = {"H_left": H_left, "H_right": H_right}
        pickle.dump(d, output, pickle.HIGHEST_PROTOCOL)


    while True:
        # Capture frame-by-frame
        retA, imageA = leftCap.read()
        retB, imageB = centerCap.read()
        retC, imageC = rightCap.read()

        if retA and retB and retC:
            shape = imageA.shape
            imgA = np.zeros((shape[0], 3 * shape[1], 3), dtype=np.uint8)
            imgA[:, 0:shape[1]] = imageA

            imgB = np.zeros((shape[0], 3 * shape[1], 3), dtype=np.uint8)
            imgB[:, shape[1]:2 * shape[1]] = imageB

            imgC = np.zeros((shape[0], 3 * shape[1], 3), dtype=np.uint8)
            imgC[:, 2 * shape[1]:] = imageC

            left = cv2.warpPerspective(imgA, H_left, (imgA.shape[1], imgA.shape[0]), flags=cv2.INTER_LINEAR)
            left = scale(left, dz_left)

            right = cv2.warpPerspective(imgC, H_right, (imgC.shape[1], imgC.shape[0]), flags=cv2.INTER_LINEAR)
            right = scale(right, dz_right)

            result = np.zeros_like(imgB)
            result[:, :shape[1]] = left[:, :shape[1]]
            result[:, 2 * shape[1]:] = right[:, 2 * shape[1]:]
            result[:, shape[1]:2 * shape[1]] = imageB
            cv2.imshow("Stitched", result)

            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break


"""
COMMANDS:
A/D - translate left/right
W/S - translate up/down
Q/E - rotate OY left/right
R/T - rotate OZ left/right
Z/X - zoom out/in
SPACE - switch camera
ESC - after cameras are calibrated, start video
Q - exit program when video running
"""