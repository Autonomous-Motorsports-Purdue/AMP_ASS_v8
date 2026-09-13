import numpy as np
import cv2
from math import factorial
from skimage.measure import ransac
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle


class Curve_fit():
    
    def __init__(self):
        self.rollingBottomCropMeans = [1] * 5
        self.rollingAvg = 1
        self.savedCount = 1
        pass

    def run(self, lanes, track):
        
        waypoint, curve = self.track(track)
        return waypoint, curve
    
    # def mst_clustering_largest_component(self, binary_image, distance_threshold=10, cut_num=0):
    #     # Step 1: Extract coordinates of foreground pixels
    #     points = np.column_stack(np.nonzero(binary_image))
        
    #     if len(points) == 0:
    #         return np.zeros_like(binary_image, dtype=np.uint8)  # empty image

    #     # Step 2: Compute distance matrix
    #     dist_matrix = distance_matrix(points, points)

    #     # Step 3: Compute MST
    #     mst = minimum_spanning_tree(dist_matrix)

    #     # Step 4: Convert MST to a graph and cut edges longer than threshold
    #     mst = mst.toarray()
    #     # count number of long edges cut
    #     cut = np.sum(mst > distance_threshold)
    #     mst[mst > distance_threshold] = 0  # cut long edges
    #     # Cut cut_num largest edges
    #     # cut only up to cut_num edges
    #     if cut > cut_num:
    #         cut_num = cut - cut_num
        
    #     if cut_num > 0:
    #         mst_flat = mst.flatten()
    #         indices = np.argpartition(mst_flat, -cut_num)[-cut_num:]
    #         mst_flat[indices] = 0
    #         mst = mst_flat.reshape(mst.shape)

    #     # Step 5: Use connected components on the modified MST
    #     graph = csr_matrix((mst + mst.T) > 0)  # symmetrize
    #     n_components, labels = connected_components(graph)

    #     # Step 6: Find largest cluster
    #     unique, counts = np.unique(labels, return_counts=True)
    #     largest_cluster_label = unique[np.argmax(counts)]

    #     # Step 7: Mask the original image
    #     mask = labels == largest_cluster_label
    #     largest_cluster_coords = points[mask]

    #     # Step 8: Create output image
    #     output_image = np.zeros_like(binary_image, dtype=np.uint8)
    #     output_image[largest_cluster_coords[:, 0], largest_cluster_coords[:, 1]] = 1

    #     return output_image

    def ransac(self, image, left):
        ys, xs = np.nonzero(image)
        points = np.column_stack([xs, ys])  # or [ys, xs], be consistent in fit/evaluate
        if len(points) < 7:
            print("Not enough points to fit a curve")
            return np.zeros_like(image, dtype=np.uint8)
        # 2. run RANSAC
        # seed is lowest point in the image
        seed = points[np.argmin(points[:, 1])]
        # if there are no points in the left quarter of the image and left is True, return empty image
        if left and np.sum(points[:, 0] < image.shape[1] * 5 // 6) == 0:
            print("No points in left quarter of the image")
            return np.zeros_like(image, dtype=np.uint8)
        # if there are no points in the right quarter of the image and left is False, return empty image
        if not left and np.sum(points[:, 0] > image.shape[1] // 6) == 0:
            print("No points in right quarter of the image")
            return np.zeros_like(image, dtype=np.uint8)
        
        xsMean = np.mean(xs)
        xsSD = np.std(xs)
        # print("xsMean: " + str(xsMean) + ", xsSD: " + str(xsSD))
        
        if False:
            for i in range(0, len(points)):
                if i >= len(points):
                    break
                if abs(abs(points[i, 0]) - abs(xsMean)) >= xsSD * 0:
                    # print(points[i, 0])
                    points = np.delete(points, i, axis=0)
                    i -= 1
               
        if True:
            dbscan = DBSCAN(eps=50, min_samples=5) # TODO: TUNE THIS
            dbscan.fit(points)
            
            # pca = PCA(n_components=2).fit(points)
            # pca_2d = pca.transform(points)
            output = []
            
            for i in range(0, len(dbscan.labels_)):
                if dbscan.labels_[i] == 0:
                    output.append([points[i, 0], points[i, 1]])
            
            newPts = np.asarray(output, dtype=np.int32)
            
            print(type(newPts))
            # print(newPts == points)
            # diff = newPts == points
            # for i in range(len(newPts == points)):
            #     if (diff[i].all() != True):
            #         print(str(newPts[i]) + " - " + str(points[i]))
            #     pass
            # cv2.imshow("dbscan", newPts)
            points = newPts
        
        best_model, inliers = ransac(
            data=points,
            model_class=lambda: BezierRansacModel(seed_pt=seed),
            min_samples=5,          # need at least degree+1 points to fit
            residual_threshold=10.0,        # pixels: tune as needed
            max_trials=len(points) // 3
        )

        # 3. recover the inlier points and make your “clean” mask
        inlier_pts = points[inliers].squeeze()
        # new_inlier_pts = []
        # for point in inlier_pts:
        #     print(point, point.shape)
        #    _ = point
        #     try:
        #         if point[0] > image.shape[1] or point[1] > image.shape[0]:
        #            pass
        #         else:
        #            new_inlier_pts.append(point)
        #     except Exception as e:
        #        print(inlier_pts)
        #        print("point", point)
        #        print(e)
        # inlier_pts = np.array(new_inlier_pts)
        #mask = (
        #    (inlier_pts[:, 0] < image.shape[1]) &
        #    (inlier_pts[:, 1] < image.shape[0])
        #)
        #inlier_pts = inlier_pts[mask]
        clean_mask = np.zeros_like(image, dtype=np.uint8)
        #print(clean_mask.shape)
        #inlier_pts = np.array([i for i in inlier_pts if (i[0] < image.shape[1] and i[1] < image.shape[0])])
        #print(inlier_pts)
        clean_mask[inlier_pts[:,1], inlier_pts[:,0]] = 1

        return clean_mask

    def get_waypoints(self, mid):
        # Get the waypoints from the mid bezier curve
        # select the middle point, if not below horizon line, then go down one and select again until below horizon line
        select = 1 * len(mid) // 2 
        top = mid[select]
        while top[1] > 380:
            select += 1
            top = mid[select]
            if select >= 39:
                return mid[39]
        return top
        bottom = mid[1 * len(mid) // 2]
        # take whichever has the more extreme x deviation
        center_x = 320
        diff_bot = (center_x - bottom[0])**2
        diff_top = (center_x - top[0])**2
        if diff_bot > diff_top:
            return bottom
        return top
        

    def track(self, img):
        # set top 50 pixels to black
        img[0:50,:] = 0
        
        # set bottom n pixels to black if their mean is >0.9
        bottomCrop = -130
        bottomCropThreshold = 0
        bottomCropBorderThreshold = 0.7
        rollingAvgThreshold = 0.98
        cropMean = float(np.mean(img[bottomCrop:img.shape[0], :])) / 255
        cropMeanBorder = float(np.mean(img[bottomCrop-5:bottomCrop, :])) / 255
        self.rollingBottomCropMeans.append(cropMean)
        self.rollingBottomCropMeans.pop(0)
        self.rollingAvg = np.mean(self.rollingBottomCropMeans)
        
        # print(self.rollingBottomCropMeans)
        # print(self.rollingAvg)
        
        # if (cropMean > bottomCropThreshold and cropMeanBorder > bottomCropBorderThreshold) or self.rollingAvg > rollingAvgThreshold:
        #     img[bottomCrop:, :] = 0
        # else:
        #     # print("cropMean: " + str(cropMean))
        #     # print("cropMeanBorder: " + str(cropMeanBorder))
        #     pass
            
        
        kernel5 = np.ones((5,5),np.uint8)
        kernel3 = np.ones((3,3),np.uint8)
        # erode = cv2.erode(img, kernel5, iterations = 5)
        # img = cv2.dilate(erode, kernel5, iterations = 5)

        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel5, iterations = 5)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel5, iterations = 2)
        #img = cv2.GaussianBlur(img, (5, 5), 0)


        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        
        if len(contours) == 0:
            print("No contours found")
            return np.array([320, 180]), np.zeros_like(img)

        # Select the largest contour
        c = max(contours, key = cv2.contourArea)

        # Draw the largest contour
        x,y,w,h = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M["m00"] != 0:  # To avoid division by zero
            contour_center_x = M["m10"] / M["m00"]
            contour_center_y = M["m01"] / M["m00"]
        else:
            contour_center_x = contour_center_y = 0
         
        rect = np.intp(cv2.boxPoints(cv2.minAreaRect(c)))
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [rect], -1, (255), -1)
        img = cv2.bitwise_and(img, mask)

        mask = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel3, iterations = 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations = 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel3, iterations = 2)

        #img = cv2.bitwise_and(img, mask)
        img = mask

        sobelLeft = cv2.Sobel(img, cv2.CV_8UC1, 1, 0, ksize=1)
        sobelLeft[:, :2] = 0
        
        with open('ransacDumps/sobelLeft' + str(self.savedCount) + '.pkl', 'wb') as f:
            pickle.dump(sobelLeft, f)
        ransac_left = self.ransac(sobelLeft, True)

        inverse = cv2.bitwise_not(img)
        sobelRight = cv2.Sobel(inverse, cv2.CV_8UC1, 1, 0, ksize=1)
        sobelRight[:, -2:] = 0
        
        with open('ransacDumps/sobelRight' + str(self.savedCount) + '.pkl', 'wb') as f:
            pickle.dump(sobelLeft, f)

        ransac_right = self.ransac(sobelRight, False)

        self.savedCount += 1

        clustered = cv2.bitwise_or(ransac_right, ransac_left)
        clustered = clustered.astype(np.uint8) * 255
        normal = cv2.bitwise_or(sobelLeft, sobelRight)
        normal = normal.astype(np.uint8) * 255
        difference = cv2.absdiff(normal, clustered)
        difference = cv2.dilate(difference, kernel3, iterations = 1)
        clustered = cv2.dilate(clustered, kernel3, iterations = 1)
        # Turn clusterd into an rgb image with difference as red channel
        clustered = cv2.cvtColor(clustered, cv2.COLOR_GRAY2BGR)

        clustered[:,:,0] = difference * 255

        cv2.imshow("clustered", clustered)
        
        

        leftCurve = get_bezier_curve(sobelLeft)
        rightCurve = get_bezier_curve(sobelRight)

        if len(np.nonzero(ransac_left)[0]) == 0:
            print("No left curve found")
            y_vals = np.linspace(img.shape[0] // 2, img.shape[0] - 1, 40, dtype=np.uint32)
            cluster_curve_left = np.column_stack((np.zeros(40), y_vals))
        else:
            cluster_curve_left = get_bezier_curve(ransac_left)
        
        if len(np.nonzero(ransac_right)[0]) == 0:
            print("No right curve found")
            y_vals = np.linspace(img.shape[0] // 2, img.shape[0] - 1, 40, dtype=np.uint32)
            cluster_curve_right = np.column_stack((np.full(40, img.shape[1] - 1), y_vals))
        else:
            cluster_curve_right = get_bezier_curve(ransac_right)

        cluster_curve_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        cluster_curve_image[:,:, 0] = img
        cv2.polylines(cluster_curve_image, [np.int32(cluster_curve_left)], isClosed=False, color=(255,255,0), thickness=2)
        cv2.polylines(cluster_curve_image, [np.int32(cluster_curve_right)], isClosed=False, color=(0,255,0), thickness=2)

        cluster_mid = (cluster_curve_left + cluster_curve_right) / 2
        cv2.polylines(cluster_curve_image, [np.int32(cluster_mid)], isClosed=False, color=(0,0,255), thickness=2)
        waypoint = self.get_waypoints(cluster_mid)
        # draw centroid on image
        cv2.circle(cluster_curve_image, (int(contour_center_x), int(contour_center_y)), 5, (155, 0, 255), -1)
        # Draw the waypoint on the curve image
        cv2.circle(cluster_curve_image, (int(waypoint[0]), int(waypoint[1])), 5, (0,255,255), -1)
        #draw horizon line
        cv2.line(cluster_curve_image, (0, int(img.shape[0] * 0.5)), (img.shape[1], int(img.shape[0] * 0.5)), (255,255,255), 2)
        #cv2.imshow("cluster_curve", cluster_curve_image)
        img_x, img_y = img.shape
        sobel = np.zeros((img_x, img_y, 3), np.uint8)
        sobel[:,:,0] = sobelLeft
        # Add sobel right as green channel
        sobel[:,:,1] = sobelRight
        cv2.imshow("sobel", sobel)
        curves_image = np.zeros_like(sobel)
        cv2.polylines(curves_image, [np.int32(leftCurve)], isClosed=False, color=(255,0,0), thickness=2)
        cv2.polylines(curves_image, [np.int32(rightCurve)], isClosed=False, color=(0,255,0), thickness=2)
        cv2.imshow("curves", curves_image)
        return np.array(waypoint), cluster_curve_image
    
class BezierRansacModel:
    def __init__(self, seed_pt=None):
        self.control_points = None
        self.seed = seed_pt

    def estimate(self, data):
        # data: (M,2) array of sample points
        if self.seed is not None:
            # data is an (m,2) array of x,y
            if not any((data == self.seed).all(axis=1)):
                return False   # reject this sample
        # otherwise fit as before
        self.control_points = get_bezier(data)
        return True

    def residuals(self, data):
        # for each data point, compute its closest distance to the fitted curve
        ts = np.linspace(0, 1, 500)                      # dense sampling
        curve_pts = plot_bezier(ts, self.control_points)  # (500,2)
        # compute minimal Euclidean distance from each data point to any curve sample
        d = np.linalg.norm(data[:, None, :] - curve_pts[None, :, :], axis=2)
        return np.min(d, axis=1)  # shape (M,)
    

def get_bezier_curve(line):
    points = np.transpose(np.nonzero(line))[0::8]
    control_points = np.array(get_bezier(points))
    t = np.linspace(0, 1, 40)
    curve = plot_bezier(t, control_points)
    curve = np.flip(curve, axis=1)
    return curve

def plot_bezier(t, cp):
    """
    Plots a bezier curve.
    t is the time values for the curve.
    cp is the control points of the curve.
    return is a tuple of the x and y values of the curve.
    """
    cp = np.array(cp)
    num_points, d = np.shape(cp)   # Number of points, Dimension of points
    num_points = num_points - 1
    curve = np.zeros((len(t), d))
    
    for i in range(num_points+1):
        # Bernstein polynomial
        val = comb(num_points,i) * t**i * (1.0-t)**(num_points-i)
        curve += np.outer(val, cp[i])
    
    return curve

def get_bezier(points):
    """
    Returns the control points of a bezier curve.
    """
    num_points = len(points)

    x, y = points[:,0], points[:,1]

    # bezier matrix for a cubic curve
    bezier_matrix = np.array([[-1, 3, -3, 1,], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])
    bezier_inverse = np.linalg.inv(bezier_matrix)

    normalized_length = normalize_path_length(points)

    points_matrix = np.zeros((num_points, 4))

    for i in range(num_points):
        points_matrix[i] = [normalized_length[i]**3, normalized_length[i]**2, normalized_length[i], 1]

    points_transpose = points_matrix.transpose()
    square_points = np.matmul(points_transpose, points_matrix)

    square_inverse = np.zeros_like(square_points)

    if (np.linalg.det(square_points) == 0):
        print("Uninvertible matrix")
        square_inverse = np.linalg.pinv(square_points)
    else:
        square_inverse = np.linalg.inv(square_points)

    # solve for the solution matrix
    solution = np.matmul(np.matmul(bezier_inverse, square_inverse), points_transpose)

    # solve for the control points
    control_points_x = np.matmul(solution, x)
    control_points_y = np.matmul(solution, y)

    return list(zip(control_points_x, control_points_y))

def normalize_path_length(points):
        """
        Returns a list of the normalized path length of the points.
        """
        path_length = [0]
        x, y = points[:,0], points[:,1]

        # calculate the path length
        for i in range(1, len(points)):
            path_length.append(np.sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2) + path_length[i - 1])
        
        # normalize the path length
        # computes the percentage of path length at each point
        pct_len = []
        for i in range(len(path_length)):
            if (path_length[i] == 0):
                pct_len.append(0.01)
                continue
            pct_len.append(path_length[i] / path_length[-1])
        
        return pct_len

def comb(n, k):
    """
    Returns the combination of n choose k.
    """
    return factorial(n) / factorial(k) / factorial(n - k)
