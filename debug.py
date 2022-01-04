import cv2
import time

cv2.namedWindow("test")

img_counter = 0
frame_width = 640
frame_height = 480
frame = cv2.imread("image_end.png")
scale_percent = 220  # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
size = (len(frame), len(frame[0]))
image_array=[]
result= cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, size)
while True:
    try:
        time.sleep(0.1)
        cv2.waitKey(50)
        frame = cv2.imread("image_start.png")
        # resize image
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("start", frame)
        frame = cv2.imread("image_end.png")
        # resize image
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        #result.write(frame)
        image_array.append(frame)
        cv2.imshow("actual", frame)
        frame = cv2.imread("A_star_plot.png")
        cv2.imshow("path", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    except:
        print("file not found")
print(2)
for i in range(len(image_array)):
    result.write(image_array[i])
result.release()
cv2.destroyAllWindows()
print(len(image_array))