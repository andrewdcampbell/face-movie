# USAGE: python face-movie/main.py (-morph | -average) -images IMAGES [-td TD] [-pd PD] [-fps FPS] -out OUT

from scipy.spatial import Delaunay
from PIL import Image
from face_morph import morph_seq, warp_im
from subprocess import Popen, PIPE
import argparse
import numpy as np
import dlib
import os
import cv2
import time

########################################
# FACIAL LANDMARK DETECTION CODE
########################################

PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)

def get_boundary_points(shape):
    h, w = shape[:2]
    boundary_pts = [
        (1,1), (w-1,1), (1, h-1), (w-1,h-1), 
        ((w-1)//2,1), (1,(h-1)//2), ((w-1)//2,h-1), ((w-1)//2,(h-1)//2)
    ]
    return np.array(boundary_pts)

def get_landmarks(im):
    rects = DETECTOR(im, 1)
    if len(rects) == 0 and len(DETECTOR(im, 0)) > 0:
        rects = DETECTOR(im, 0)

    if len(rects) == 0:
        return None

    target_rect = rects[0] 
    if len(rects) > 1:
        target_rect = prompt_user_to_choose_face(im, rects)

    landmarks = np.array([(p.x, p.y) for p in PREDICTOR(im, target_rect).parts()])
    landmarks = np.append(landmarks, get_boundary_points(im.shape), axis=0)
    return landmarks

def prompt_user_to_choose_face(im, rects):
    im = im.copy()
    h, w = im.shape[:2]
    for i in range(len(rects)):
        d = rects[i]
        x1, y1, x2, y2 = d.left(), d.top(), d.right()+1, d.bottom()+1
        cv2.rectangle(im, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=5)
        cv2.putText(im, str(i), (d.center().x, d.center().y),
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=1.5,
                    color=(255, 255, 255),
                    thickness=5)

    DISPLAY_HEIGHT = 650
    resized = cv2.resize(im, (int(w * DISPLAY_HEIGHT / float(h)), DISPLAY_HEIGHT))
    cv2.imshow("Multiple faces", resized); cv2.waitKey(1)
    target_index = int(input("Please choose the index of the target face: "))
    cv2.destroyAllWindows(); cv2.waitKey(1)
    return rects[target_index] 

########################################
# VISUALIZATION CODE FOR DEBUGGING
########################################    

def draw_triangulation(im, landmarks, triangulation):
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.triplot(landmarks[:,0], landmarks[:,1], triangulation, color='blue', linewidth=1)
    plt.axis('off')
    plt.show()

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0], point[1])
        cv2.putText(im, str(idx+1), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255))
        cv2.circle(im, pos, 3, color=(255, 0, 0))
    cv2.imwrite("landmarks.jpg", im)


########################################
# MAIN DRIVER FUNCTIONS
########################################

def average_images(out_name):
    avg_landmarks = sum(LANDMARK_LIST) / len(LANDMARK_LIST)
    triangulation = Delaunay(avg_landmarks).simplices

    warped_ims = [
        warp_im(np.float32(IM_LIST[i]), LANDMARK_LIST[i], avg_landmarks, triangulation) 
        for i in range(len(LANDMARK_LIST))
    ]

    average = (1.0 / len(LANDMARK_LIST)) * sum(warped_ims)
    average = np.uint8(average)

    cv2.imwrite(out_name, average)

def morph_images(duration, fps, pause_duration, out_name):
    first_im = cv2.cvtColor(IM_LIST[0], cv2.COLOR_BGR2RGB)
    h = max(first_im.shape[:2])
    w = min(first_im.shape[:2])    

    command = ['ffmpeg', 
        '-y', 
        '-f', 'image2pipe', 
        '-r', str(fps), 
        '-s', str(h) + 'x' + str(w), 
        '-i', '-', 
        '-c:v', 'libx264', 
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', 
        '-pix_fmt', 'yuv420p', 
        out_name,
    ]         

    p = Popen(command, stdin=PIPE)

    pause_frames = int(fps * pause_duration)
    fill_frames(Image.fromarray(first_im), pause_frames, p)

    for i in range(len(IM_LIST) - 1):
        print("Morphing {} to {}".format(IM_FILES[i], IM_FILES[i+1]))
        last_frame = morph_pair(i, i+1, duration, fps, out_name, p)
        fill_frames(last_frame, pause_frames, p)

    p.stdin.close()
    p.wait()


def morph_pair(idx1, idx2, duration, fps, out_name, stream):
    """
    For a pair of images, produce a morph sequence with the given duration
    and fps to be written to the provided output stream.
    """
    im1 = IM_LIST[idx1]
    im2 = IM_LIST[idx2]

    im1_landmarks = LANDMARK_LIST[idx1]
    im2_landmarks = LANDMARK_LIST[idx2]

    total_frames = int(duration * fps)

    if im1_landmarks is None or im2_landmarks is None:
        print("No faces found, performing cross-dissolve between {} and {}"
            .format(IM_FILES[idx1], IM_FILES[idx2]))
        cross_dissolve(total_frames, im1, im2, stream)

    else:
        average_landmarks = (im1_landmarks + im2_landmarks) / 2  

        triangulation = Delaunay(average_landmarks).simplices
        # draw_triangulation(im2, average_landmarks, triangulation)

        h, w = im1.shape[:2]
        morph_seq(total_frames, im1, im2, im1_landmarks, im2_landmarks, 
            triangulation.tolist(), (w, h), out_name, stream)
    return Image.fromarray(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))

# TODO: less janky way of filling frames?
def fill_frames(im, num, p):
    for _ in range(num):
        im.save(p.stdin, 'JPEG')

def cross_dissolve(total_frames, im1, im2, p):
    for j in range(total_frames):
        alpha = j / (total_frames - 1)    
        blended = (1.0 - alpha) * im1 + alpha * im2
        im = Image.fromarray(cv2.cvtColor(np.uint8(blended), cv2.COLOR_BGR2RGB))
        im.save(p.stdin, 'JPEG')

if __name__ == "__main__":
    start_time = time.time()
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("-morph", help="Create morph sequence", action='store_true')
    group.add_argument("-average", help="Create average face", action='store_true')
    ap.add_argument("-images", help="Directory of input images", required=True)
    ap.add_argument("-td", type=float, help="Transition duration (in seconds)", default=3.0)
    ap.add_argument("-pd", type=float, help="Pause duration (in seconds)", default=0.0)
    ap.add_argument("-fps", type=int, help="Frames per second", default=25)
    ap.add_argument("-out", help="Output file name", required=True)
    args = vars(ap.parse_args())

    MORPH = args["morph"]
    IM_DIR = args["images"]
    FRAME_RATE = args["fps"]
    DURATION = args["td"]
    PAUSE_DURATION = args["pd"]
    OUTPUT_NAME = args["out"]

    valid_formats = [".jpg", ".jpeg", ".png"]
    get_ext = lambda f: os.path.splitext(f)[1].lower()

    # Constraints on input images (for morphing):
    # - Must all have same dimension
    # - Must have clear frontal view of a face (there may be multiple)
    # - Filenames must be in lexicographic order of the order in which they are to appear

    IM_FILES = [f for f in os.listdir(IM_DIR) if get_ext(f) in valid_formats]
    IM_FILES = sorted(IM_FILES, key=lambda x: x.split('/'))
    assert len(IM_FILES) > 0, "No valid images found in {}".format(IM_DIR)

    IM_LIST = [cv2.imread(IM_DIR + '/' + f, cv2.IMREAD_COLOR) for f in IM_FILES]
    print("Detecting landmarks...")
    LANDMARK_LIST = [get_landmarks(im) for im in IM_LIST]
    print("Starting...")

    if MORPH:
        morph_images(DURATION, FRAME_RATE, PAUSE_DURATION, OUTPUT_NAME)
    else:
        average_images(OUTPUT_NAME)

    elapsed_time = time.time() - start_time
    print("Time elapsed: {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
