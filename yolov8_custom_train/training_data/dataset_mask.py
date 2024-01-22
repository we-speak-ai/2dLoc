import cv2
import os
import numpy as np
from sklearn.cluster import DBSCAN

path = '/mnt/data/wespeakai/projects/2Dloc/src/'

def find_template(target_image_path, template_image_path):
    # Read the images
    target_image = cv2.imread(target_image_path)
    template = cv2.imread(template_image_path, 0)  # 0 to read image in grayscale

    # Convert the target image to grayscale
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(target_gray, template, cv2.TM_CCOEFF_NORMED)

    # Find the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Draw a rectangle around the matched region
    top_left = max_loc
    h, w = template.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(target_image, top_left, bottom_right, 255, 2)

    # Display the result
    cv2.imwrite(f'{path}/result.jpg', target_image)


def flann(target_image_path, template_image_path):
    img1 = cv2.imread(target_image_path,cv2.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv2.imread(template_image_path,cv2.IMREAD_GRAYSCALE) # trainImage
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    good_points = []
    good_points_idxs = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good_points.append(np.array(kp1[m.queryIdx].pt))
            good_points_idxs.append(i)
            matchesMask[i] = [1,0]
    good_points = np.array(good_points)

    db = DBSCAN(eps=100, min_samples=3).fit(good_points)
    labels = db.labels_
    for i, matchesMask_idx in enumerate(good_points_idxs):
        if i > len(labels):
            assert('no way')
        if labels[i] < 0:
            matchesMask[matchesMask_idx] = [0, 0]
    
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    cv2.imwrite(f'{path}/result_match.jpg', img3)


    good_points = good_points[labels >= 0]
    good_points = np.array(good_points, dtype=np.float32)
    good_points = good_points.reshape(-1, 1, 2)

    return good_points

def crop_around_matched_kps(image_path, points):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Calculate the bounding rectangle
    x, y, w, h = cv2.boundingRect(points)

    # Calculate the center of the bounding rectangle
    center_x, center_y = x + w / 2, y + h / 2

    # Double the size of the rectangle
    size = 6 * max(w, h)
    new_w, new_h = size, size

    # Compute the new bounding box coordinates
    start_x = int(max(center_x - new_w / 2, 0))
    start_y = int(max(center_y - new_h / 2, 0))
    end_x = int(min(center_x + new_w / 2, image.shape[1]))
    end_y = int(min(center_y + new_h / 2, image.shape[0]))

    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]
    if cropped_image.size != 0:
        # Save or display the cropped image
        cv2.imwrite(f'{path}/result_cropped.jpg', cropped_image)
        return cropped_image
    return None

def rect_finder(file_name):
    #image = image.copy()
    image = cv2.imread(f"{path}/result_cropped.jpg")
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    iw, ih, id = image.shape

    scale_factor = 1
    if iw < 100 or ih < 100:
        scale_factor = 3
        gray = cv2.resize(gray, (iw*scale_factor, ih*scale_factor), interpolation=cv2.INTER_CUBIC)
    
    blur_size = int(iw / 40 * 5)
    if not blur_size % 2:
        blur_size += 1
    #print('blur_size=', blur_size)
    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    cv2.imwrite(f'{path}/result_gray_cropped.jpg', gray)

    # increase contract
    contrast_factor = 2.5
    gray = cv2.convertScaleAbs(gray, alpha=contrast_factor, beta=0)
    cv2.imwrite(f'{path}/result_gray_cropped_contract.jpg', gray)

    # Find the darkest point
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
    min_val = max(min_val, 1.0) 

    # Define your threshold value
    threshold_value = (max_val - min_val) / 2

    # Apply the threshold
    _, binary_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Save or display the binary image
    cv2.imwrite(f'{path}/result_mask_binary.jpg', binary_image)

    # Edge detection
    edges = cv2.Canny(binary_image, 50, 150)
    cv2.imwrite(f'{path}/result_edges.jpg', edges)

    # Finding contours
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours and approximate the polygonal curves
    for cnt in contours:
        # Calculate the perimeter of the contour
        peri = cv2.arcLength(cnt, True)
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        # If the approximated polygon has 4 vertices, it could be a rectangle
        if len(approx) == 4:
            # Check if the polygon is convex
            if cv2.isContourConvex(approx):
                # downscale result 
                approx = np.round(approx / scale_factor).astype(int)

                # Draw the contour on the image
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
                cv2.imwrite(f'{path}/result_rect.jpg', image)

                # Create an empty mask of zeros (black) with the same dimensions as original image
                mask = np.ones_like(image, dtype=np.uint8) * 255
                cv2.drawContours(mask, [approx], -1, color=(0, 0, 0), thickness=cv2.FILLED)
                cv2.imwrite(f'{path}/result_mask.jpg', mask)

                area = cv2.contourArea(approx)
                if area < 20:
                    print(f'rect finder failed for {file_name}')
                    return None
                return mask
    return None

def generate_mask_all():
    input_dir = f"{path}data/video_imgs/video_imgs/"
    template_image = f"{path}data/template.png"

    input_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.jpg'):
            input_files.append(file)

    print(path)
    print(input_files)
    input_files = ['color_ISO1000_EXP11000_v2_010.jpg']
    # mask = rect_finder(None)
    # exit()

    for input_file in input_files:
        input_image = f'{input_dir}{input_file}'
        print(input_image)
        good_points = flann(input_image, template_image)
        cropped_image = crop_around_matched_kps(input_image, good_points)
        if cropped_image is None:
            continue
        mask = rect_finder()

        # save images
        if mask is not None:
            dir_path = f'{path}data/train/bw_code'
            base_name = os.path.basename(input_image).rsplit('.', 1)[0]
            # cropped img
            cv2.imwrite(f'{dir_path}/images/{base_name}.png', cropped_image)
            # mask
            cv2.imwrite(f'{dir_path}/masks/{base_name}.png', mask)

def generate_mask_from_bbox():
    input_dir = f"{path}/data/train_ds/"
    input_dirs = []
    for dirpath, dirnames, _ in os.walk(input_dir):
        if not dirnames:  # This means there are no subdirectories in dirpath
            input_dirs.append(dirpath)

    for input_dir in input_dirs:
        
        basename = input_dir.split('/')[-2:]
        basename = '_'.join(basename)
        
        input_files = []
        for file in os.listdir(input_dir):
            if file.endswith('.jpg'):
                input_files.append(file)

        #input_files = ['frame_34.jpg']
        # mask = rect_finder(None)
        # exit()

        for input_file in input_files:
            # input_dir = '/mnt/data/wespeakai/projects/2Dloc/src//data/train_ds/mono/mono_ISO100_EXP10000/'
            # input_file = 'frame_254.jpg'
            input_image = f'{input_dir}/{input_file}'

            # read image
            image = cv2.imread(input_image)
            # read mask
            bbox_file = input_image.split('.')[0] + '.txt'
            with open(bbox_file, 'r') as file:
                # Read the first line from the file
                line = file.readline()
            # Split the line into a list of strings, each representing a number
            numbers_str = line.split()
            # Convert each string into an integer
            x, y, w, h = [int(num) for num in numbers_str]
            # extend bbox
            x -= w/2
            y -= h/2
            w *= 2
            h *= 2
            x = max(x, 0)
            y = max(y, 0)
            x, y, w, h = int(x), int(y), int(w), int(h)

            cropped_image = image[y:y+h, x:x+w]
            cv2.imwrite(f'{path}/result_cropped.jpg', cropped_image)

            # crop img
            mask = rect_finder(input_dir + '/' + input_file)

            # save images
            if mask is not None:
                dir_path = f'{path}data/train/bw_code'
                os.makedirs(f'{dir_path}/images', exist_ok=True)
                os.makedirs(f'{dir_path}/masks', exist_ok=True)
                base_name = basename + '_' + os.path.basename(input_image).rsplit('.', 1)[0]
                # cropped img
                cv2.imwrite(f'{dir_path}/images/{base_name}.png', cropped_image)
                # mask
                cv2.imwrite(f'{dir_path}/masks/{base_name}.png', mask)
                
generate_mask_from_bbox()