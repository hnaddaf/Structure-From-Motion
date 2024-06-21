import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import glob
import CAMERAMAT 

cameraMatrixL=CAMERAMAT.cameraMat()
print (cameraMatrixL)


images=[]
images_d= sorted(glob.glob('C:/Users/husse/Desktop/MSc. Robotcs/Sensing and Preciption/corseWork/images/Test/?.png'))
for i in range(len(images_d)):
    img=cv.imread(images_d[i])
    
    images.append(img)

def findKeypoints(img1,img2):

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

    
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
    good_matches_for_draw = [[m] for m in good_matches]

    img = cv.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good_matches_for_draw, None, flags=2)

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
 
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

points_img_1=[]
points_img_2=[]
for i in range(len(images)-1):

    points1,points2,imgToDraw=findKeypoints(images[i],images[i+1])
    points_img_1.append(points1)
    points_img_2.append(points2)
    plt.figure(i+1)
    plt.imshow(imgToDraw)
plt.show()



def estimate_Fmatrix(img1_pts,img2_pts):

    x1 = img1_pts[:,0]
    y1 = img1_pts[:,1]
    x1dash = img2_pts[:,0]
    y1dash = img2_pts[:,1]
    A = np.zeros((len(x1),9))
    for i in range(len(x1)):
        A[i] = np.array([x1dash[i]*x1[i],x1dash[i]*y1[i],x1dash[i], y1dash[i]*x1[i],y1dash[i]*y1[i],y1dash[i],x1[i],y1[i],1])

    #taking SVD of A for estimation of F
    U, S, V = np.linalg.svd(A,full_matrices=True)
    F_est = V[-1, :]
    F_est = F_est.reshape(3,3)

    # Enforcing rank 2 for F
    ua,sa,va = np.linalg.svd(F_est,full_matrices=True)
    sa = np.diag(sa)

    sa[2,2] = 0
   

    F = np.dot(ua,np.dot(sa,va))
    return F

F_numaric=[]
E_from_F1=[]
E_from_OCV=[]

for i in range(len(images)-1):

    F1=estimate_Fmatrix(points_img_1[i],points_img_2[i])

    F_numaric.append(F1)

    E1=cameraMatrixL.T @ F1 @ cameraMatrixL

    points1_norm = cv.undistortPoints(np.expand_dims(points_img_1[i], axis=1), cameraMatrixL, None)
    points2_norm = cv.undistortPoints(np.expand_dims(points_img_2[i], axis=1), cameraMatrixL, None)
    E4, mask = cv.findEssentialMat(points1_norm, points2_norm, focal=1.0, pp=(0, 0), method=cv.RANSAC, prob=0.999, threshold=1.0)
    E_from_F1.append(E1)

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

R_1_calculated=[]
T_1_calculated=[]
R_4_calculated=[]
T_4_calculated=[]
R_4_L=[]
T_4_L=[]

for i in range(len(images)-1):


    R4, T4=get_RTset(E_from_OCV[i])
    R1, T1=get_RTset(E_from_F1[i])

    _, r4, t4, mask = cv.recoverPose(E_from_OCV[i], points_img_1[i], points_img_2[i], cameraMatrix=cameraMatrixL)

    R_1_calculated.append(R1)
    T_1_calculated.append(T1)

    R_4_calculated.append(R4)
    T_4_calculated.append(T4)


    R_4_L.append(r4)
    T_4_L.append(t4)


p1=[points_img_1[0][1,0],points_img_1[0][1,1],1]
p2=[points_img_2[0][1,0],points_img_2[0][1,1],1]
print(p1)
print(p2)
p3=R_1_calculated[0]@p2+T_1_calculated[0].T
print('COMPAIR')
print(p3)



def point_triangulation(k,pt1,pt2,R1,C1,R2,C2):
    points_3d = []

    I = np.identity(3)
    C1 = C1.reshape(3,1)
    C2 = C2.reshape(3,1)

    #calculating projection matrix P = K[R|T]
    P1 = np.dot(k,np.dot(R1,np.hstack((I,-C1))))
    P2 = np.dot(k,np.dot(R2,np.hstack((I,-C2))))
  
    #homogeneous coordinates for images
    xy = np.hstack((pt1,np.ones((len(pt1),1))))
    xy_cap = np.hstack((pt2,np.ones((len(pt1),1))))

    
    p1,p2,p3 = P1
    p1_cap, p2_cap,p3_cap = P2

    #constructing contraints matrix
    for i in range(len(xy)):
        A = []
        x = xy[i][0]
        y = xy[i][1]
        x_cap = xy_cap[i][0]
        y_cap = xy_cap[i][1] 
        
        A.append((y*p3) - p2)
        A.append((x*p3) - p1)
        
        A.append((y_cap*p3_cap)- p2_cap)
        A.append((x_cap*p3_cap) - p1_cap)

        A = np.array(A).reshape(4,4)

        _, _, v = np.linalg.svd(A)
        x_ = v[-1,:]
        x_ = x_/x_[-1]
        # x_ =x_[:3]
        points_3d.append(x_)


    return np.array(points_3d)

def linear_triangulation(R_Set,T_Set,pt1,pt2,k):
    R1_ = np.identity(3)
    T1_ = np.zeros((3,1))
    points3d = point_triangulation(k,pt1,pt2,R1_,T1_,R_Set,T_Set)
    points3d=points3d[:,0:3]
    return points3d



transformation_matrix=[]
rotation=R_4_calculated
translation=T_4_calculated
print('traslation = ',translation)
R_M=[]
T_V=[]
R_M.append(rotation[0])
T_V.append(translation[0])

transformation_matrix1 = np.zeros((4, 4)) 
transformation_matrix1[:3, :3] = rotation[0] 
transformation_matrix1[:3, 3] = translation[0].flatten()  
transformation_matrix1[3, :] = [0, 0, 0, 1] 

transformation_matrix.append(transformation_matrix1)

def GetHemoMatrix (transformation_matrix1,r,t):
    
    transformation_matrix2 = np.zeros((4, 4))   
    transformation_matrix2[:3, :3] = r 
    transformation_matrix2[:3, 3] = t.flatten()  
    transformation_matrix2[3, :] = [0, 0, 0, 1]

    transformation_matrix=transformation_matrix1@transformation_matrix2
    return transformation_matrix

for i in range(len(images)-2):
    transformation_matrix_new=GetHemoMatrix (transformation_matrix[i],rotation[i+1],translation[i+1])    
    transformation_matrix.append(transformation_matrix_new)
    R=transformation_matrix[i+1][:3, :3]
    T=transformation_matrix[i+1][:3, 3]
    R_M.append(R)
    T_V.append(T)
l=0
Save_points=[]
for i in range(len(images)-1):
    points_3d = linear_triangulation(rotation[i],translation[i],points_img_1[i],points_img_2[i],cameraMatrixL)
    Save_points.append(points_3d)
    l=l+len(points_3d)
   
All_points=np.zeros((l, 3))
s=0


for i in range(len(Save_points)):
    All_points[s:s+len(Save_points[i]), :]=Save_points[i]
    s=len(Save_points[i])


pc1=np.array([[0],[0],[1],[1]])

for i in range(len(transformation_matrix)):
    hp=transformation_matrix@pc1


def plot_camera_positions(transformations):
    # Start at the origin
    current_position = np.array([0, 1, 0, 1])  # Homogeneous coordinates
    positions = [current_position[:3]]  # Extract x, y, z

    # Apply each transformation
    for t in transformations:
        current_position = np.dot(t, current_position)
        positions.append(current_position[:3])  # Extract x, y, z after transformation
        current_position = np.array([0, 1, 0, 1])  # Homogeneous coordinates
    # Convert positions to numpy array for easy slicing
    positions = np.array(positions)
    X=[]
    Y=[]
    X.append([0.0])
    Y.append([1.0])
    for i in range(len(translation)):
        x=translation[i][0]
        y=translation[i][1]+1
        X.append(x)
        Y.append(y)
    # Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(positions[:, 0], positions[:, 1], '-o', label='Estimated Camera Position')
    plt.plot(np.array([0,-0.8]), np.array([1,1.17]), '-*', label='Actual Camera Position')
    plt.plot([0], [0], 'ro', label='object')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Camera Positions')
    plt.legend()
    plt.axis('equal')  # Equal scaling for x and y axes
    plt.grid(True)
    plt.show()

plot_camera_positions(transformation_matrix)
#print(points_img_1)