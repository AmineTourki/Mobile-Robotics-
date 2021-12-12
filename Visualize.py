import cv2
# This file is not used in the principal implementation, only for visualization purposes.
cv2.namedWindow("test")

img_counter = 0

while True:
    try:
        frame = cv2.imread("image_start.png")
        cv2.imshow("start", frame)
        frame = cv2.imread("image_end.png")
        cv2.imshow("actual", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    except:
        print("file not found")
print(2)
cv2.destroyAllWindows()