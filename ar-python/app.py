import cv2
import numpy as np

# ウェブカメラからArUcoマーカーを検出し、その位置姿勢を推定するサンプルコード
import cv2.aruco as aruco

# Placeholder values
cameraMatrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float) # fx,fy,cx,cy
distCoeffs = np.zeros((4,1)) # distortion coefficients

# Load the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Define the dimensions of the cube
cube_size = 0.1

# Define the 3D coordinates of the cube vertices
cube_points = np.array([
    [-cube_size/2, -cube_size/2, 0],             #下面の左下の頂点
    [cube_size/2, -cube_size/2, 0],              #下面の右下の頂点
    [cube_size/2, cube_size/2, 0],               #下面の右上の頂点
    [-cube_size/2, cube_size/2, 0],              #下面の左上の頂点
    [-cube_size/2, -cube_size/2, cube_size],     #上面の左下の頂点
    [cube_size/2, -cube_size/2, cube_size],      #上面の右下の頂点
    [cube_size/2, cube_size/2, cube_size],       #上面の右上の頂点
    [-cube_size/2, cube_size/2, cube_size]       #上面の左上の頂点
])

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the grayscale frame
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict)

    # If any markers are detected
    if ids is not None:
        # Draw bounding boxes around the detected markers
        aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Estimate the pose of each marker
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs)
        
        # Draw the cube on each marker
        for rvec, tvec in zip(rvecs, tvecs):
            # Convert rotation vector to rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)

            # Project the 3D cube points onto the image plane
            cube_points_2d, _ = cv2.projectPoints(cube_points, rvec, tvec, cameraMatrix, distCoeffs)

            # Draw the cube by connecting the projected points and fill each face with a different color
            colors = [
                (255, 0, 0),   # Blue
                (0, 255, 0),   # Green
                (0, 0, 255),   # Red
                (255, 255, 0), # Cyan
                (255, 0, 255), # Magenta
                (0, 255, 255)  # Yellow
            ]

            # Draw the cube by connecting the projected points and fill each face with a different color
            for i in range(4):
                pt1 = tuple(map(int, cube_points_2d[(i+1)%4].ravel()))
                pt2 = tuple(map(int, cube_points_2d[i].ravel()))
                pt3 = tuple(map(int, cube_points_2d[i+4].ravel()))
                pt4 = tuple(map(int, cube_points_2d[(i+1)%4+4].ravel()))
                
                # Draw the lines connecting the points
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                cv2.line(frame, pt2, pt3, (0, 255, 0), 2)
                cv2.line(frame, pt3, pt4, (0, 255, 0), 2)
                cv2.line(frame, pt4, pt1, (0, 255, 0), 2)
                
                # Fill the face with a different color
                cv2.fillPoly(frame, [np.array([pt1, pt2, pt3, pt4])], colors[i])
                
            # Draw the top face of the cube
            pt1 = tuple(map(int, cube_points_2d[4].ravel()))
            pt2 = tuple(map(int, cube_points_2d[5].ravel()))
            pt3 = tuple(map(int, cube_points_2d[6].ravel()))
            pt4 = tuple(map(int, cube_points_2d[7].ravel()))

            # Draw the lines connecting the points
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.line(frame, pt2, pt3, (0, 255, 0), 2)
            cv2.line(frame, pt3, pt4, (0, 255, 0), 2)
            cv2.line(frame, pt4, pt1, (0, 255, 0), 2)

            # Fill the top face with a different color
            cv2.fillPoly(frame, [np.array([pt1, pt2, pt3, pt4])], colors[5])
                
    # Display the frame with overlaid markers
    cv2.imshow("Frame with ArUco markers", frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()