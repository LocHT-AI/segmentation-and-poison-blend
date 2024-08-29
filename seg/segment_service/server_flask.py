import os
from flask import Flask, request, render_template, send_from_directory
import cv2
import json 
from logs.log_handler import logger
from ultralytics import YOLO
from config.loader import MODEL_SEG,MODEL_POSE
from PIL import Image
import matplotlib.pyplot as plt
from skimage.draw import polygon
from pyamg.gallery import poisson
from scipy.sparse import csr_matrix
from pyamg import ruge_stuben_solver

app = Flask(__name__)

# Định nghĩa thư mục lưu trữ hình ảnh tải lên và kết quả
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

model_seg=YOLO(MODEL_SEG)
model_pose=YOLO(MODEL_POSE)
import cv2
import numpy as np

def rgbToGrayMat(imgPth):
    gryImg = Image.open(imgPth).convert('L')
    return np.asarray(gryImg)

def cropImageByLimits(src, minRow, maxRow, minCol, maxCol):
    r, g, b = src
    r = r[minRow: maxRow, minCol: maxCol]
    g = g[minRow: maxRow, minCol: maxCol]
    b = b[minRow: maxRow, minCol: maxCol]
    return r, g, b
def splitImageToRgb(imgPth):
    r, g, b = Image.Image.split(Image.open(imgPth))
    return np.asarray(r), np.asarray(g), np.asarray(b)

def topLeftCornerOfSrcOnDst(dstImgPth, srcShp,center):
    gryDst = rgbToGrayMat(dstImgPth)
    # logger.debug("vlluon")
    center = [[2.95,      1.19]] # Corrected syntax    
    if len(center) < 1:
        center = np.asarray([[gryDst.shape[1] // 2, gryDst.shape[0] // 2]]).astype(int)
    elif len(center) > 1:
        center = np.asarray([center[0]])
    corner = [center[0][1] - srcShp[0] // 2, center[0][0] - srcShp[1] // 2]
    return keepSrcInDstBoundaries(corner, gryDst.shape, srcShp)

def cropDstUnderSrc(dstImg, corner, srcShp):
    dstUnderSrc = dstImg[
                  corner[0]:corner[0] + srcShp[0],
                  corner[1]:corner[1] + srcShp[1]]
    return dstUnderSrc

def fixCoeffUnderBoundaryCondition(coeff, shape):
    shapeProd = np.prod(np.asarray(shape))
    arangeSpace = np.arange(shapeProd).reshape(shape)
    arangeSpace[1:-1, 1:-1] = -1
    indexToChange = arangeSpace[arangeSpace > -1]
    for j in indexToChange:
        coeff[j, j] = 1
        if j - 1 > -1:
            coeff[j, j - 1] = 0
        if j + 1 < shapeProd:
            coeff[j, j + 1] = 0
        if j - shape[-1] > - 1:
            coeff[j, j - shape[-1]] = 0
        if j + shape[-1] < shapeProd:
            coeff[j, j + shape[-1]] = 0
    return coeff

def setBoundaryCondition(b, dstUnderSrc):
    b[1, :] = dstUnderSrc[1, :]
    b[-2, :] = dstUnderSrc[-2, :]
    b[:, 1] = dstUnderSrc[:, 1]
    b[:, -2] = dstUnderSrc[:, -2]
    b = b[1:-1, 1: -1]
    return b

def laplacian(array):
    return poisson(array.shape, format='csr') * csr_matrix(array.flatten()).transpose().toarray()

def constructConstVector(mask, mixedGrad, dstUnderSrc, srcLaplacianed, srcShp):
    dstLaplacianed = laplacian(dstUnderSrc)
    
    b = np.reshape(
        (1 - mixedGrad) * mask * np.reshape(srcLaplacianed, srcShp) +
        mixedGrad * mask * np.reshape(dstLaplacianed, dstUnderSrc.shape) +
        (mask - 1) * (-1) * np.reshape(dstLaplacianed, dstUnderSrc.shape),
        dstUnderSrc.shape
    )
    
    return setBoundaryCondition(b, dstUnderSrc)


def constructCoefficientMat(shape):
    a = poisson(shape, format='lil')
    a = fixCoeffUnderBoundaryCondition(a, shape)
    return a

def buildLinearSystem(mask, srcImg, dstUnderSrc, mixedGrad):
    srcLaplacianed = laplacian(srcImg)
    b = constructConstVector(mask, mixedGrad, dstUnderSrc, srcLaplacianed, srcImg.shape)
    a = constructCoefficientMat(b.shape)
    return a, b


def solveLinearSystem(a, b, bShape):
    multiLevel = ruge_stuben_solver(csr_matrix(a))
    x = np.reshape((multiLevel.solve(b.flatten(), tol=1e-10)), bShape)
    x[x < 0] = 0
    x[x > 255] = 255
    return x

def keepSrcInDstBoundaries(corner, gryDstShp, srcShp):
    for idx in range(len(corner)):
        if corner[idx] < 1:
            corner[idx] = 1
        if corner[idx] > gryDstShp[idx] - srcShp[idx] - 1:
            corner[idx] = gryDstShp[idx] - srcShp[idx] - 1
    return corner

def blend(dst, patch, corner, patchShape, blended):
    mixed = dst.copy()
    mixed[corner[0]:corner[0] + patchShape[0], corner[1]:corner[1] + patchShape[1]] = patch
    blended.append(Image.fromarray(mixed))
    return blended

def poissonAndNaiveBlending(mask, corner, srcRgb, dstRgb, mixedGrad):
    poissonBlended = []
    naiveBlended = []
    for color in range(3):
        src = srcRgb[color]
        dst = dstRgb[color]
        dstUnderSrc = cropDstUnderSrc(dst, corner, src.shape)
        a, b = buildLinearSystem(mask, src, dstUnderSrc, mixedGrad)
        x = solveLinearSystem(a, b, b.shape)
        poissonBlended = blend(dst, x, (corner[0] + 1, corner[1] + 1), b.shape, poissonBlended)
        cropSrc = mask * src + (mask - 1) * (- 1) * dstUnderSrc
        naiveBlended = blend(dst, cropSrc, corner, src.shape, naiveBlended)
    return poissonBlended, naiveBlended

def seg(input_path):
    detect = model_seg(input_path, conf=0.4)
    for result in detect:
        bbox=result.boxes.xyxy
        logger.debug(bbox)
        width, height = get_bbox_size(bbox)
        for mask in result.masks.xy:

            mask_array = mask  # Access the entire array directly
            # logger.debug(mask)
            # masks.append(mask_array)
    # logger.debug(mask_array)
    img = rgbToGrayMat(input_path)

    if len(mask_array) < 3:
        minRow, minCol = (0, 0)
        maxRow, maxCol = img.shape
        mask = np.ones(img.shape)
    else:
        mask_array = np.fliplr(mask_array)
        inPolyRow, inPolyCol = polygon(tuple(mask_array[:, 0]), tuple(mask_array[:, 1]), img.shape)
        minRow, minCol = (np.max(np.vstack([np.floor(np.min(mask_array, axis=0)).astype(int).reshape((1, 2)), (0, 0)]),
                                 axis=0))
        maxRow, maxCol = (np.min(np.vstack([np.ceil(np.max(mask_array, axis=0)).astype(int).reshape((1, 2)), img.shape]),
                                 axis=0))
        mask = np.zeros(img.shape)
        mask[inPolyRow, inPolyCol] = 1
        mask = mask[minRow: maxRow, minCol: maxCol]
    return width,height,mask, minRow, maxRow, minCol, maxCol

def get_bbox_size(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height



def get_hand_pose(wrist_pose, elbow_pose):
    # dis_wrist_elbow = dis_of_2_points(wrist_pose, elbow_pose)
    arm_direction = get_direct_arm(wrist_pose, elbow_pose)

    hand_pose_x = wrist_pose[0] - 1 / 4 * arm_direction[0]
    hand_pose_y = wrist_pose[1] - 1 / 4 * arm_direction[1]

    return int(hand_pose_x), int(hand_pose_y)

def get_direct_arm(wrist_pose, elbow_pose):
    return elbow_pose[0] - wrist_pose[0], elbow_pose[1] - wrist_pose[1]

def pose(input_path):
    results = model_pose(input_path, conf=0.4)
    # keypoint = []
    # hands_pose=[]
    hand_pose = None
    for result in result:
        bbox=result.boxes.xyxy
        logger.debug(bbox)
        width, height = get_bbox_size(bbox)
    for xy in results[0].keypoints.xy:
        # keypoint=xy
        try:
            left_wrist_pose = (int(xy.tolist()[10][0]), int(xy.tolist()[10][1]))
            left_elbow_pose = (int(xy.tolist()[8][0]), int(xy.tolist()[8][1]))
            left_hand_pose = get_hand_pose(left_wrist_pose, left_elbow_pose)

            right_wrist_pose = (int(xy.tolist()[9][0]), int(xy.tolist()[9][1]))
            right_elbow_pose = (int(xy.tolist()[7][0]), int(xy.tolist()[7][1]))
            right_hand_pose = get_hand_pose(right_wrist_pose, right_elbow_pose)
            hand_pose = (right_elbow_pose, right_wrist_pose, right_hand_pose)
            break
        except:
            pass
    # logger.debug(keypoint)
    # hand_pose=keypoint[10]
    # hand_pose=np.array(hand_pose)
    # hands_pose.append(hand_pose)
    # 
    # hand_pose=np.array(hand_pose)
    logger.debug(hand_pose)
    return hand_pose,width, height

def mergeSaveShow(splitImg, ImgTtl):
    merged = Image.merge('RGB', tuple(splitImg))
    merged.save(ImgTtl + '.png')  

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file1 = request.files['image1']
        file2 = request.files['image2']
        if file1 and file2:
            # Lưu tệp hình ảnh tải lên vào thư mục tạm thời
            filename1 = os.path.join(app.config['UPLOAD_FOLDER'], 'input1.png')
            filename2 = os.path.join(app.config['UPLOAD_FOLDER'], 'input2.png')
            
            file1.save(filename1)
            file2.save(filename2)
            # Thực hiện phân đoạn
            # results = seg(filename1)            
            width_1, height_1, mask, *maskLimits = seg(filename1)
            point,width, height=pose(filename2)
            rgb = splitImageToRgb(filename1)
            srcRgbCropped = cropImageByLimits(rgb, *maskLimits)
            # logger.debug(mask)
            rgb_v1 = splitImageToRgb(filename2)
            logger.debug("..")
            corner = topLeftCornerOfSrcOnDst(filename2, srcRgbCropped[0].shape,point)
            logger.debug("...")
            # Hiển thị kết quả trên trang result.html
            result, result_2 = poissonAndNaiveBlending(mask, corner, srcRgbCropped, rgb_v1, 0.3)
            logger.debug("....")
            mergeSaveShow(result,'result')
            mergeSaveShow(result_2,'result_2')
            return render_template('result.html', input_image1=filename1, input_image2=filename2, segment_result1=result, segment_result2=result_2)

    return render_template('index.html')

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
