import cv2
import numpy as np
# This Function Will Detect The Lines From The Canny Edge Detection
def canny(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny
def ROI(img):
    height = img.shape[0]
    polygons = np.array([[(200,height),(1100,height),(550,250)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image
def displayLines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.ravel()
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),20)
    return line_image
def sameSizeLine(image,lineParameters):
    slop,intercept = lineParameters
    y1 = image.shape[0] # It Will The Height Of The Image
    y2 = int(y1*(3/5) )
    x1 = int((y1-intercept)/slop)
    x2 = int((y2-intercept)/slop)
    return np.array([x1,y1,x2,y2])
def averageSlopIntercept(image,lines):
    left_lines = []
    right_lines = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slop = parameters[0]
        intercept = parameters[1]
        if slop < 0:
            left_lines.append((slop,intercept))
        else:
            right_lines.append((slop,intercept))
    leftLinesAverage =  np.average(left_lines,axis=0)
    rightLinesAverage = np.average(right_lines,axis=0)
    left_line = sameSizeLine(image,leftLinesAverage)
    right_line = sameSizeLine(image,rightLinesAverage)
    return np.array([left_line,right_line])
# img = cv2.imread('./test_image.jpg')
# lane_image = np.copy(img)
# canny = canny(lane_image)
# ROI(canny)
# mask = ROI(canny)
# lines = cv2.HoughLinesP(mask,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
# average_lines = averageSlopIntercept(lane_image,lines)
# line_image = displayLines(lane_image,average_lines)
# line_image = cv2.addWeighted(lane_image,0.8,line_image,1,0)
# cv2.imshow("Result",line_image)
# if cv2.waitKey(0) & 0xFF == ord('q'):
#     exit()
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
cap = cv2.VideoCapture('test2.mp4')
while(cap.isOpened()):
    ret,frame = cap.read()
    frame1 = rescale_frame(frame,50)
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    mask = ROI(canny)
    lines = cv2.HoughLinesP(mask,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    average_lines = averageSlopIntercept(frame,lines)
    line_image = displayLines(frame,average_lines)
    line_image = cv2.addWeighted(frame,0.8,line_image,1,0)
    cv2.imshow("Result",line_image)
    cv2.imshow("original Video ",frame1)
    if cv2.waitKey(35) & 0xFF == ord('q'):
        exit()