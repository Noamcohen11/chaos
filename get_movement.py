#################################################################
# Project : Chaos experiment.py
# WRITER : Noam Cohen
#################################################################

##############################################################################
#                                   imports                                  #
##############################################################################

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##############################################################################
#                                   Constants                                #
##############################################################################
FPS = 60
# Read the video from the file
video = cv2.VideoCapture(
    "/Users/noamcohen/Google Drive/My Drive/כאוס/G&I/Recordings/4.mov"
)

# the range of blue color in HSV
HSV_DICT = {
    "red": (np.array([0, 100, 100]), np.array([13, 255, 255])),
    # "red": (np.array([13, 100, 100]), np.array([22, 132, 255])),
    "green": (np.array([29, 90, 38]), np.array([58, 255, 255])),
    "orange": (np.array([0, 143, 134]), np.array([66, 255, 255])),
}

RGB_DICT = {"green": (0, 255, 0), "red": (0, 0, 255), "orange": (0, 128, 229)}


# orange_center = (580, 328)
##############################################################################
#                                   Functions                                #
##############################################################################


def distance(x1, y1, x2, y2):
    """
    Calculate distance between two points
    """
    dist = math.sqrt(
        (np.float(x2) - np.float(x1)) ** 2 + (np.float(y2) - np.float(y1)) ** 2
    )
    return dist


def angle_calc(p1, p2):
    color1_x, color1_y = p1
    color2_x, color2_y = p2
    hypotenuse = distance(color1_x, color1_y, color2_x, color2_y)
    horizontal = distance(color1_x, color1_y, color2_x, color1_y)
    vertical = distance(color2_x, color1_y, color2_x, color2_y)
    # angle = np.arcsin(horizontal / hypotenuse)

    if color2_y > color1_y and color2_x > color1_x:
        return np.arcsin(horizontal / hypotenuse)
    elif color2_y > color1_y and color2_x < color1_x:
        return np.arcsin(-horizontal / hypotenuse)
    elif color2_y < color1_y and color2_x < color1_x:
        return np.arcsin(-vertical / hypotenuse) - math.pi / 2
    elif color2_y < color1_y and color2_x > color1_x:
        return np.arcsin(vertical / hypotenuse) + math.pi / 2

    return np.arcsin(horizontal / hypotenuse)


def max_angle_calc(p1, p2):
    color1_x, color1_y = p1
    color2_x, color2_y = p2
    color1_xx = color1_x + 30
    color1_yy = color1_y - 30
    hypotenuse = distance(color1_xx, color1_yy, color2_x, color2_y)
    horizontal = distance(color1_xx, color1_yy, color2_x, color1_yy)
    vertical = distance(color2_x, color1_yy, color2_x, color2_y)
    # angle = np.arcsin(horizontal / hypotenuse)

    if color2_y > color1_y and color2_x > color1_x:
        return np.arcsin(horizontal / hypotenuse)
    elif color2_y > color1_y and color2_x < color1_x:
        return np.arcsin(-horizontal / hypotenuse)
    elif color2_y < color1_y and color2_x < color1_x:
        return np.arcsin(-vertical / hypotenuse) - math.pi / 2
    elif color2_y < color1_y and color2_x > color1_x:
        return np.arcsin(vertical / hypotenuse) + math.pi / 2

    return np.arcsin(horizontal / hypotenuse)


def get_colored_circle(hsv, frame_gau_blur, color):
    # getting the range of red color in frame
    min_hsv, max_hsv = HSV_DICT[color]
    hsv_range = cv2.inRange(hsv, min_hsv, max_hsv)
    res = cv2.bitwise_and(frame_gau_blur, frame_gau_blur, mask=hsv_range)
    s_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    canny_edge = cv2.Canny(s_gray, 50, 240)

    # applying HoughCircles
    circles = cv2.HoughCircles(
        canny_edge,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=4,
        param2=8,
        minRadius=0,
        maxRadius=10,
    )
    if type(circles) != np.ndarray:
        # applying HoughCircles
        circles = cv2.HoughCircles(
            canny_edge,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=10,
            param1=4,
            param2=8,
            minRadius=10,
            maxRadius=20,
        )

    # cv2.imshow("gray", blue_s_gray)
    # cv2.imshow(
    #     "canny", canny_edge
    # )

    return circles


def circle_center(frame, circles, color, f_center=None):
    if type(circles) == np.ndarray:
        circles = np.uint16(np.around(circles))
        if len(circles[0, :]) == 1:
            x, y, r = circles[0, :][0]
            # drawing on detected circle and its center
            cv2.circle(frame, (x, y), r, RGB_DICT[color], 2)
            cv2.circle(frame, (x, y), 2, RGB_DICT[color], 3)
            return (x, y)
        elif len(circles[0, :]) > 1 and f_center is not None:
            min_distance = 100
            best_circle = None
            for circle in circles[0, :]:
                dist = distance(circle[0], circle[1], f_center[0], f_center[1])
                if dist < min_distance:
                    min_distance = dist
                    best_circle = circle
            if best_circle is not None:
                x, y, r = best_circle
                # drawing on detected circle and its center
                cv2.circle(frame, (x, y), r, RGB_DICT[color], 2)
                cv2.circle(frame, (x, y), 2, RGB_DICT[color], 3)
                return (x, y)
    return None


def dimension_changer(green_angles, red_angles, green_vs, red_vs):

    minimized_red_angles = []
    minimized_green_vs = []
    minimized_red_vs = []
    for i in range(1, len(green_angles) - 1):
        if (
            abs(green_angles[i]) < abs(green_angles[i - 1])
            and abs(green_angles[i]) < abs(green_angles[i + 1])
            and (
                (green_angles[i - 1] > 0 and green_angles[i + 1] < 0)
                or (green_angles[i - 1] < 0 and green_angles[i + 1] > 0)
            )
        ):
            minimized_red_angles.append(red_angles[i])
            minimized_green_vs.append(green_vs[i])
            minimized_red_vs.append(red_vs[i])

    return minimized_red_angles, minimized_green_vs, minimized_red_vs


import csv


def plot_movement(theta_error, theta2, omega, t):
    """Plots the graph"""

    start = next(i for i in range(len(t)) if t[i] > 15)
    t = t[start::]
    t = [x - t[0] for x in t]
    theta_error = theta_error[start::]
    theta2 = theta2[start::]
    omega = omega[start::]
    import scipy.signal as signal

    plt.grid()
    plt.show()
    num_of_images = 15
    print(num_of_images)

    # Initialize empty lists to store data
    highlimit = 2
    # Open the file and skip the first 129 rows
    theta_1_time = []
    theta1 = []

    with open(
        "/Users/noamcohen/Google Drive/My Drive/כאוס/G&I/DATA/9.csv",
        "r",
    ) as file:
        reader = csv.reader(file)
        i = 0
        for row in reader:
            if i < 3:
                i += 1
                continue
            if float(row[0]) < 41:
                continue
            theta_1_time.append(float(row[0]))
            theta1.append(float(row[1]))

    theta_1_time = [x - theta_1_time[0] for x in theta_1_time]
    start = 0
    for i in range(num_of_images):
        figure, axis = plt.subplots(2, 2)
        if i == 0:
            IMAGE_LEN = 11
        else:
            IMAGE_LEN = 12
        loop_t = t[FPS * IMAGE_LEN * i : FPS * IMAGE_LEN * (i + 1)]
        loop_theta = theta2[FPS * IMAGE_LEN * i : FPS * IMAGE_LEN * (i + 1)]
        error = theta_error[FPS * IMAGE_LEN * i : FPS * IMAGE_LEN * (i + 1)]
        w = np.linspace(0.01, 15, FPS)
        pgram = signal.lombscargle(
            loop_t,
            loop_theta,
            w,
            normalize=True,
        )

        max_ti = next(
            i for i in range(len(theta_1_time)) if theta_1_time[i] > loop_t[-1]
        )
        # Compute the Fourier transform of theta1
        thet = theta1[start:max_ti]
        theta1_fft = np.fft.fft(thet)
        ti = theta_1_time[start:max_ti]
        start = max_ti

        # Get the frequencies of the Fourier transform in Hz
        # Assume uniform sampling
        freq = np.fft.fftfreq(len(ti), d=ti[2] - ti[1])
        freq_hz = freq

        mask = (freq_hz >= 0) & (freq_hz <= highlimit)
        freq_hz_masked = freq_hz[mask]
        theta1_fft_masked = (((theta1_fft[mask]) * 0.002) ** 2) ** (1 / 2)

        axis[1, 0].plot(
            loop_t,
            loop_theta,
            "o",
            markersize=3,
            color="blue",
        )
        axis[1, 0].errorbar(
            loop_t,
            loop_theta,
            yerr=error,
            fmt="o",
            markersize=3,
            color="blue",
        )
        axis[1, 0].set_title(r"$\theta_2$")
        axis[1, 1].plot(w, pgram)
        axis[1, 1].set_title(r"$\theta_2$ FTT")
        axis[0, 0].plot(
            ti,
            thet,
            "o",
            markersize=3,
            color="blue",
        )
        axis[0, 0].set_title(r"$\theta_1$")
        axis[0, 1].plot(freq_hz_masked, theta1_fft_masked)
        axis[0, 1].set_title(r"$\theta_1$ FTT")
        axis[0, 1].set_ylim([0, 1])
        axis[1, 1].set_ylim([0, 1])
        for ax in axis.flat:
            ax.set_xlabel(xlabel="time [sec]", fontsize=11)
            ax.set_ylabel("angle [rad]", fontsize=11)
        plt.tight_layout()
        plt.show()


def main():

    # Collect points:
    green_angles = []
    red_angles = []
    green_vs = []
    error_reds = []
    red_vs = []

    time = []
    timer = 0

    pre_green_angle = None
    pre_red_angle = None
    get_zero_point = True

    bad_frame_counter = 1

    red_center = None
    green_center = None

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)

    # Below VideoWriter object will create
    # a frame of above defined The output
    # is stored in 'filename.avi' file.
    result = cv2.VideoWriter(
        "filename.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, size
    )
    while True:
        # Read the next frame
        ret, frame = video.read()
        # If we have reached the end of the video, break the loop
        if not ret:
            break

        # blurring the frame that's captured
        frame_gau_blur = cv2.GaussianBlur(frame, (3, 3), 0)
        # converting BGR to HSV
        hsv = cv2.cvtColor(frame_gau_blur, cv2.COLOR_BGR2HSV)

        # Get the point in which the Pendulum is hang.
        if get_zero_point:
            orange_circle = get_colored_circle(hsv, frame_gau_blur, "orange")
            orange_center = circle_center(frame, orange_circle, "orange")
            if orange_center:
                get_zero_point = False

        # Get circles and find their center.
        red_circle = get_colored_circle(hsv, frame_gau_blur, "red")
        green_circle = get_colored_circle(hsv, frame_gau_blur, "green")
        red_center = circle_center(frame, red_circle, "red", red_center)
        green_center = circle_center(
            frame, green_circle, "green", green_center
        )

        # Get the angle between the cicles.
        if green_center and orange_center:
            green_angle = angle_calc(orange_center, green_center)
            cv2.putText(
                frame,
                "theta1=" + str(green_angle),
                (300, 150),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                RGB_DICT["orange"],
                2,
            )
        red_angle = 0
        if green_center and red_center and orange_center:
            if (
                distance(
                    orange_center[0],
                    orange_center[1],
                    red_center[0],
                    red_center[1],
                )
                < 25
            ):
                bad_frame_counter += 1
                timer += 1 / FPS
                continue

            red_angle = angle_calc(green_center, red_center)
            cv2.putText(
                frame,
                "theta2=" + str(red_angle),
                (300, 200),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                RGB_DICT["red"],
                2,
            )
            error_red = abs(
                max_angle_calc(green_center, red_center) - red_angle
            )

            # Gather data for graph
            green_angles.append(green_angle)
            error_reds.append(error_red)
            red_angles.append(red_angle)
            if pre_green_angle != None and pre_red_angle != None:
                red_vs.append(
                    (red_angle - pre_red_angle) * FPS / bad_frame_counter
                )
                green_vs.append(
                    (green_angle - pre_green_angle) * FPS / bad_frame_counter
                )
                cv2.putText(
                    frame,
                    "v1="
                    + str(
                        (green_angle - pre_green_angle)
                        * FPS
                        / bad_frame_counter
                    ),
                    (300, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    RGB_DICT["green"],
                    2,
                )
                cv2.putText(
                    frame,
                    "v2="
                    + str(
                        (red_angle - pre_red_angle) * FPS / bad_frame_counter
                    ),
                    (300, 100),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    RGB_DICT["green"],
                    2,
                )
            else:
                red_vs.append(0)
                green_vs.append(0)

            pre_green_angle = green_angle
            pre_red_angle = red_angle
            time.append(timer)
            bad_frame_counter = 1

        else:
            bad_frame_counter += 1

        timer += 1 / FPS

        result.write(frame)
        result.write(frame)
        cv2.imshow("circles", frame)
        # check for q to quit program with 5ms delay
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    # clean up our resources
    result.release()
    video.release()
    cv2.destroyAllWindows()

    # Plot 2d graph
    # plot_movement(error_reds, red_angles, red_vs, time)


if __name__ == "__main__":
    main()
