import cv2

# class Image_Publisher():
#     def __init__(self):
#         self.cap = cv2.VideoCapture(0)
#         self.frame = None
#     def run(self):
#         print("running")
#         ret, self.frame = self.cap.read()
#         if self.frame is not None:
#             cv2.imshow("Image", self.frame)
#             return self.frame
            

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
        # if not ret:
        # print("Error: Could not read frame.")
        # break

    # Display the frame
    cv2.imshow("Webcam Feed", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()