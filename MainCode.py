import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import glob
import CAMERAMAT 

cameraMatrixL=CAMERAMAT.cameraMat()# From the file CAMERAMAT, the function calculate the camera intrinsic matrix
print (cameraMatrixL)

directory='images/SfM Photos/*.png'# please change the diroctory to the right one in your PC, 
images=[]
images_d= sorted(glob.glob(directory)) 
for i in range(len(images_d)):
    img=cv.imread(images_d[i])
    
    images.append(img)
# Save the images in the (images) set
    
def findKeypoints(img1,img2): # This function use sift to find the main featurs and BFMatcher to find the matched featurs and gives the image where the matched featurs is linked to draw and the location of the matched featurs

    sift = cv.SIFT_create()
    # Detect SIFT keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []

    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)

    # Get the good matches and stor them in (good_matches) list
            
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
    # Find the 2D position of the matched featurs in bth of the mathced photos 
    good_matches_for_draw = [[m] for m in good_matches]
    img = cv.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good_matches_for_draw, None, flags=2)
    # Create an image contain both of the compaired images and draw lines between the matched points
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # agest the colors of the pixels to RGB
    return points1,points2,img_rgb

def draw_key_features(images):

    sift = cv.SIFT_create()
    plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        keypoints, descriptors = sift.detectAndCompute(images[i], None)
        img_with_keypoints = cv.drawKeypoints(images[i], keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        plt.figure(i+1)
        plt.imshow(cv.cvtColor(img_with_keypoints, cv.COLOR_BGR2RGB))
        plt.title('Image with SIFT Keypoints')
    plt.show()

draw_key_features(images)
# Show all inputs photos with the sift featues marked
points_img_1=[]
points_img_2=[]
for i in range(len(images)-1):

    points1,points2,imgToDraw=findKeypoints(images[0],images[i+1])
    points_img_1.append(points1)
    points_img_2.append(points2)
    # find the matched featurs between the first image and all other images and store them in the lists (points_img_1 for the location of the points in te first image) and (points_img_2 for the matched point feature with the image we comparing with)
    plt.figure(i+1)
    plt.imshow(imgToDraw)
plt.show()
#Draw all the compaired images with the maching lines


def estimate_Fmatrix(img1_pts, img2_pts):# calculate the Fundemantal matrix using svd
    # Extract x, y coordinates from both sets of points
    x1 = img1_pts[:, 0]
    y1 = img1_pts[:, 1]
    x1dash = img2_pts[:, 0]
    y1dash = img2_pts[:, 1]

    # Initialize matrix A based on the size of input points
    A = np.zeros((len(x1), 9))
    for i in range(len(x1)):
        # Populate A with elements based on the epipolar constraint equation for each point pair
        A[i] = np.array([x1dash[i]*x1[i], x1dash[i]*y1[i], x1dash[i], 
                         y1dash[i]*x1[i], y1dash[i]*y1[i], y1dash[i], 
                         x1[i], y1[i], 1])

    # Perform Singular Value Decomposition (SVD) on A to estimate the Fundamental matrix (F)
    U, S, V = np.linalg.svd(A, full_matrices=True)
    # The last row of V (or V^T) corresponds to the smallest singular value, which is used to reshape into F
    F_est = V[-1, :]
    F_est = F_est.reshape(3, 3)

    # Enforcing the rank 2 constraint on F by setting the smallest singular value to 0
    # This is done by performing SVD on the estimated F and modifying the singular values
    ua, sa, va = np.linalg.svd(F_est, full_matrices=True)
    sa = np.diag(sa)
    sa[2, 2] = 0  # Set the smallest singular value to 0

    # Recompute F using the modified singular values to ensure it has rank 2
    F = np.dot(ua, np.dot(sa, va))

    # Return the corrected Fundamental matrix
    return F

F_numaric=[]
E_from_F=[]
E_from_OCV=[]

for i in range(len(images)-1): # repeat the following comands in the numper of camera transformations ( numper of images -1)

    F=estimate_Fmatrix(points_img_1[i],points_img_2[i])
    # Find the Fundimantal Matrix
    F_numaric.append(F)
    # add the F matrix to the list F_numaric to be used later
    E1=cameraMatrixL.T @ F @ cameraMatrixL
    # Calculate the Essintial Matrix in the first Method
    points1_norm = cv.undistortPoints(np.expand_dims(points_img_1[i], axis=1), cameraMatrixL, None)
    points2_norm = cv.undistortPoints(np.expand_dims(points_img_2[i], axis=1), cameraMatrixL, None)
    E4, mask = cv.findEssentialMat(points1_norm, points2_norm, focal=1.0, pp=(0, 0), method=cv.RANSAC, prob=0.999, threshold=1.0)
     # Calculate the Essintial Matrix in the second Method (openCV)
    E_from_F.append(E1)
    E_from_OCV.append(E4)


def get_RTset(E):

    U, S, V = np.linalg.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    R = np.dot(U,np.dot(W,V))
    T = -U[:,2]
    if (np.linalg.det(R)<0):
        R=-R
        T=-T

    return R, T
# Calculate the camera position ( rotation and translation of the camera) using the algorthim in [4] n.d, "Structure from Motion," CMSC426 Computer Vision, [Online]. Available: https://cmsc426.github.io/sfm/. [Accessed 17 2 2024].

###########################################################################################################################################
###################### Store All The Rotation matrices and translation Vectors from the three diffrent methods #########################
R_1_calculated=[]
T_1_calculated=[]
R_4_calculated=[]
T_4_calculated=[]
R_4_L=[]
T_4_L=[]

for i in range(len(images)-1):


    R4, T4=get_RTset(E_from_OCV[i])
    # Camera Position, mentioned in the report as Method 1
    _, r4, t4, mask = cv.recoverPose(E_from_OCV[i], points_img_1[i], points_img_2[i], cameraMatrix=cameraMatrixL)
    # Camera Position, mentioned in the report as Method 2
    R1, T1=get_RTset(E_from_F[i])
    # Camera Position, mentioned in the report as Method 3

    R_1_calculated.append(R1)
    T_1_calculated.append(T1)

    R_4_calculated.append(R4)
    T_4_calculated.append(T4)

    R_4_L.append(r4)
    T_4_L.append(t4)

#########################################################################################################################################
######################################################### Linear Triangulation##########################################################

def point_triangulation(k, pt1, pt2, R1, C1, R2, C2, img1):
    points_3d = []

    # Identity matrix for constructing projection matrices
    I = np.identity(3)
    # Reshape camera centers (translation vector) 
    C1 = C1.reshape(3, 1)
    C2 = C2.reshape(3, 1)

    # Calculate the projection matrices for both cameras
    # Projection matrix P = K[R|T], where T is the translation vector (-RC)
    P1 = np.dot(k, np.dot(R1, np.hstack((I, -C1))))
    P2 = np.dot(k, np.dot(R2, np.hstack((I, -C2))))
  
    # Convert 2D points to homogeneous coordinates for triangulation
    xy = np.hstack((pt1, np.ones((len(pt1), 1))))
    xy_cap = np.hstack((pt2, np.ones((len(pt1), 1))))

    # Decompose projection matrices to individual rows for constructing constraint matrix
    p1, p2, p3 = P1
    p1_cap, p2_cap, p3_cap = P2

    # Constructing constraint matrix A for each pair of points and solve for 3D point
    for i in range(len(xy)):
        A = []
        x, y = xy[i, :2]
        x_cap, y_cap = xy_cap[i, :2]

        # Constraint equations derived from the cross product condition x'^T * F * x = 0
        A.append(y * p3 - p2)
        A.append(x * p3 - p1)
        A.append(y_cap * p3_cap - p2_cap)
        A.append(x_cap * p3_cap - p1_cap)

        # Perform SVD on the constraint matrix to solve for the 3D point
        A = np.array(A).reshape(4, 4)
        _, _, v = np.linalg.svd(A)
        x_ = v[-1, :]
        x_ = x_ / x_[-1]  # Convert from homogeneous to Cartesian coordinates

        points_3d.append(x_)
    
    # Extracting color information for each 2D point in the first image
    colors = []
    for point in pt1.astype(int):
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)  # Convert color space if needed
        color = img1[point[1], point[0]]  # Fetch the color at the 2D point location
        colors.append(color)

    return np.array(points_3d), np.array(colors)

def linear_triangulation(R_Set, T_Set, pt1, pt2, k, img1):
    # Identity and zero matrices for camera 1 to use as reference
    R1_ = np.identity(3)
    T1_ = np.zeros((3, 1))
    # Triangulate points and obtain their colors
    points3d, colors = point_triangulation(k, pt1, pt2, R1_, T1_, R_Set, T_Set, img1)
    points3d = points3d[:, 0:3]  # Discard the homogeneous coordinate
    return points3d, colors


#######################################################################################################################################

transformation_matrix=[]
rotation=R_4_L
translation=T_4_L
Save_points=[]
all_colors=[]
for i in range(len(images)-1):
    R1_ = np.identity(3)
    T1_ = np.zeros((3, 1))
    points_3d, colors = linear_triangulation(rotation[i],translation[i],points_img_1[i],points_img_2[i],cameraMatrixL,images[i])
    inlier_matches = np.stack((points_img_1[i], points_img_2[i]), axis=1)
    Save_points.append(points_3d)
    all_colors.append(colors)
    
D3colrs=all_colors[0]
D3points=Save_points[0]
for i in range(len(Save_points)-1):
    D3points = np.vstack((D3points, Save_points[i+1]))
    D3colrs = np.vstack((D3colrs, all_colors[i+1]))

#Get all the colors and points from all matched featurs overall photos in one array (D3points,D3colrs)


def plot(points_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(points_3d)):
        ax.scatter(points_3d[i][:,0], points_3d[i][:,1], points_3d[i][:,2])
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Scatter Plot Example')
    ax.set_xlim(-2, 2)  
    ax.set_ylim(-2, 2)  
    ax.set_zlim(-2, 2)
    plt.show()



def plotxy(Save_points):

    
    for i in range(len(Save_points)):
        plt.scatter(Save_points[i][:,0], Save_points[i][:,1])    
    plt.title('2D Points Plot')
    plt.xlabel('x')
    plt.ylabel('y')   
    plt.xlim(-2, 2)  
    plt.ylim(-2, 2)
    plt.show()

def plotxz(Save_points):

    
    for i in range(len(Save_points)):
        plt.scatter(Save_points[i][:,0], Save_points[i][:,2])    
    plt.title('2D Points Plot')
    plt.xlabel('x')
    plt.ylabel('z')  
    plt.xlim(-2, 2)  
    plt.ylim(-2, 2) 
    plt.show()

def plotyz(Save_points):
    for i in range(len(Save_points)):
        plt.scatter(Save_points[i][:,1], Save_points[i][:,2])     
    plt.title('2D Points Plot')
    plt.xlabel('y')
    plt.ylabel('z')   
    plt.xlim(-2, 2)  
    plt.ylim(-2, 2)
    plt.show()

plot(Save_points)
plotxy(Save_points)
plotxz(Save_points)
plotyz(Save_points)

def plot_camera_positions(translation):
    
    X=[]
    Y=[]
    X.append([0.0])
    Y.append([0.0])
    for i in range(len(translation)): 
        x=translation[i][0]
        y=translation[i][1]
        X.append(x)
        Y.append(y)
    # Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(X, Y, '-o', label='Estimated Camera Position')
    plt.plot([0.3], [0.3], 'ro', label='object')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Camera Positions')
    plt.legend()
    plt.axis('equal')  # Equal scaling for x and y axes
    plt.grid(True)
    plt.show()

plot_camera_positions(translation)
#Thi function plot the position of the camera at each shot starting from the orgin when the opject is at (0.3, 0.3) m in the space and the orgin is where I took the first photo


def create_point_cloud_file(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')
print("3D model saved to pointCloud.ply")

output_file = 'pointCloud.ply'

# Generate point cloud file to store the 3D points and name it pointCloud.ply
create_point_cloud_file(D3points,D3colrs, output_file)

