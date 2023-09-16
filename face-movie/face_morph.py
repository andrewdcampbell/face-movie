import numpy as np
import cv2
from PIL import Image


def warp_im(im, src_landmarks, dst_landmarks, dst_triangulation):
    # im_out = np.zeros_like(im)
    im_out = im.copy()

    for i in range(len(dst_triangulation)):
        src_tri = src_landmarks[dst_triangulation[i]]
        dst_tri = dst_landmarks[dst_triangulation[i]]
        morph_triangle(im, im_out, src_tri, dst_tri)

    return im_out


def morph_triangle(im, im_out, src_tri, dst_tri):
    # For efficiency, we crop out a rectangular region containing the triangles 
    # to warp only that small part of the image.

    # Get bounding boxes around triangles
    sr = cv2.boundingRect(np.float32([src_tri]))
    dr = cv2.boundingRect(np.float32([dst_tri]))

    # Get new triangle coordinates reflecting their location in bounding box
    cropped_src_tri = [(src_tri[i][0] - sr[0], src_tri[i][1] - sr[1]) for i in range(3)]
    cropped_dst_tri = [(dst_tri[i][0] - dr[0], dst_tri[i][1] - dr[1]) for i in range(3)]

    # Create mask for destination triangle
    mask = np.zeros((dr[3], dr[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(cropped_dst_tri), (1.0, 1.0, 1.0), 16, 0)

    # Crop input image to corresponding bounding box
    cropped_im = im[sr[1]:sr[1] + sr[3], sr[0]:sr[0] + sr[2]]

    size = (dr[2], dr[3])
    warpImage1 = affine_transform(cropped_im, cropped_src_tri, cropped_dst_tri, size)

    # Copy triangular region of the cropped patch to the output image
    im_out[dr[1]:dr[1]+dr[3], dr[0]:dr[0]+dr[2]] = \
        im_out[dr[1]:dr[1]+dr[3], dr[0]:dr[0]+dr[2]] * (1 - mask) + warpImage1 * mask


def affine_transform(src, src_tri, dst_tri, size):
    M = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    # BORDER_REFLECT_101 is good for hiding seems
    dst = cv2.warpAffine(src, M, size, borderMode=cv2.BORDER_REFLECT_101)
    return dst        


def morph_seq(total_frames, im1, im2, im1_landmarks, im2_landmarks, 
              triangulation, size, out_name, stream):

    im1 = np.float32(im1)
    im2 = np.float32(im2)

    for j in range(total_frames):
        alpha = j / (total_frames - 1)
        weighted_landmarks = (1.0 - alpha) * im1_landmarks + alpha * im2_landmarks

        warped_im1 = warp_im(im1, im1_landmarks, weighted_landmarks, triangulation)
        warped_im2 = warp_im(im2, im2_landmarks, weighted_landmarks, triangulation)

        blended = (1.0 - alpha) * warped_im1 + alpha * warped_im2

        # Convert to PIL Image and save to the pipe stream
        res = Image.fromarray(cv2.cvtColor(np.uint8(blended), cv2.COLOR_BGR2RGB))
        res.save(stream.stdin, 'JPEG')
        print(j)

    return res

