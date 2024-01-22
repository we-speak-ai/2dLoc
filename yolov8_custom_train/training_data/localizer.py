import cv2
import numpy as np
from scipy.stats import median_abs_deviation as mad
from utils import imshow


class Localizer:
    def __init__(self, marker_img_path, rough_tracker_model_dir, debug=False):
        self.debug = debug
        self.tracker = self.get_nano_tracker(rough_tracker_model_dir)

        self.marker_img_ref_size = (50, 50)
        self.marker_img = self.load_marker_img(marker_img_path, marker_img_ref_size=self.marker_img_ref_size)

        self.marker_physical_size = (47, 47)  # marker's size in mm
        self.marker_virtual_size = None
        self.marker_matcher = MarkerMatcher(marker_img_path, self.marker_img_ref_size)
        self.marker_location = None
        self.age_of_marker_location = None

    def set_virtual_marker_size(self, pix_to_mm_ratio):
        self.marker_virtual_size = (self.marker_physical_size[0] / pix_to_mm_ratio,
                                    self.marker_physical_size[1] / pix_to_mm_ratio)

    def get_marker_size(self):
        return self.marker_virtual_size

    @staticmethod
    def get_nano_tracker(rough_tracker_model_dir):
        params = cv2.TrackerNano_Params()
        params.backbone = f'{rough_tracker_model_dir}/nanotrack_backbone_sim.onnx' # an onnx file downloaded from the url displayed in (your doc)[https://docs.opencv.org/4.7.0/d8/d69/classcv_1_1TrackerNano.html]
        params.neckhead = f'{rough_tracker_model_dir}/nanotrack_head_sim.onnx'
        return cv2.TrackerNano_create(params)

    def load_marker_img(self, marker_img_path, marker_img_ref_size):
        marker_img = cv2.imread(marker_img_path)
        marker_img = cv2.resize(marker_img, marker_img_ref_size)
        return marker_img

    def find_marker_based_on_kps(self, img):
        marker_bbox, box_corners = self.marker_matcher.find_marker_on_frame(img)
        if marker_bbox is not None:
            self.marker_location = marker_bbox
            self.age_of_marker_location = 0
        return marker_bbox, box_corners

    def update_marker_location(self, img):
        roi_img, x, y = self.get_search_roi_from_previous_bbox(img, self.marker_location, bbox_age_coef=1)
        if self.debug:
            cv2.imshow('ROI', roi_img)
            cv2.waitKey()

        new_bbox, box_corners = self.find_marker_based_on_kps(roi_img)
        if new_bbox is not None:
            new_bbox = list(new_bbox)
            new_bbox[0] += x
            new_bbox[1] += y
            self.marker_location = new_bbox
            self.age_of_marker_location = 0
        elif self.marker_location is not None:
            self.age_of_marker_location +=1
        if box_corners is not None:
            box_corners[:, :, 0] += x
            box_corners[:, :, 1] += y
        return new_bbox, box_corners

    def get_search_roi_from_previous_bbox(self, img, prev_bbox=None, bbox_age_coef=0.5):
        if prev_bbox is not None:
            area_coef = (self.age_of_marker_location + 1) * bbox_age_coef
            bb = prev_bbox
            s1 = area_coef
            s2 = s1 * 2 + 1
            x, y, w, h = (int(np.max([bb[0] - s1 * bb[2], 0])),
                          int(np.max([bb[1] - s1 * bb[3], 0])), int(bb[2] * s2), int(bb[3] * s2))
            x2 = int(np.min([x + w, img.shape[1]]))
            y2 = int(np.min([y + h, img.shape[0]]))
            roi_img = img[y: y2, x: x2] # todo kivágás menjen eggyel kintebb
        else:
            x, y = 0, 0
            roi_img = img
        return roi_img, x, y





class MarkerMatcher:
    def __init__(self, marker_img_path, marker_img_ref_size, nr_kps_to_use=40, debug=False):
        self.debug = debug
        self.feature_detector = cv2.SIFT_create()
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.nr_kps_to_use = nr_kps_to_use
        self.marker_img = cv2.imread(marker_img_path)
        self.marker_img_viz = cv2.resize(self.marker_img, marker_img_ref_size)
        self.marker_img_ref_size = marker_img_ref_size
        self.marker_img_sizes = [50, 100]
        self.marker_blur_window_sizes = (0, 0), (5, 5), (7, 7)
        self.motion_blur_window_sizes = [9, 15]
        self.marker_kps, self.marker_desc = self.get_marker_keypoints_and_descriptors(self.marker_img,
                                                                                      self.marker_img_sizes,
                                                                                      self.marker_blur_window_sizes,
                                                                                      self.motion_blur_window_sizes,
                                                                                      self.marker_img_ref_size)

    def get_marker_keypoints_and_descriptors(self, marker_img,
                                             marker_img_sizes,
                                             blur_window_sizes,
                                             motion_blur_window_sizes,
                                             ref_marker_size):
        all_marker_kps, all_marker_desc = [], []
        for s in marker_img_sizes:
            m_img_ = cv2.resize(marker_img, (s, s))
            ref_s_to_s_ratio = ref_marker_size[0] / s
            for bw in blur_window_sizes:
                gb_m_img_ = cv2.GaussianBlur(m_img_, bw, 0) if bw[0] > 0 else np.array(m_img_)
                if self.debug:
                    cv2.imshow("marker img", gb_m_img_)
                    cv2.waitKey()
                marker_kps, marker_desc = self.feature_detector.detectAndCompute(gb_m_img_, None)
                for m_kps in marker_kps:
                    m_kps.pt = (m_kps.pt[0]*ref_s_to_s_ratio, m_kps.pt[1]*ref_s_to_s_ratio)
                all_marker_kps += marker_kps
                all_marker_desc.append(marker_desc)

            for mbw in motion_blur_window_sizes:
                v_img, h_img = self.create_motion_blurred_img(m_img_, kernel_size=mbw)
                v_marker_kps, v_marker_desc = self.feature_detector.detectAndCompute(v_img, None)

                if self.debug:
                    cv2.imshow("v_img", v_img)
                    cv2.imshow("h_img", h_img)
                    cv2.waitKey()
                for m_kps in v_marker_kps:
                    m_kps.pt = (m_kps.pt[0]*ref_s_to_s_ratio, m_kps.pt[1]*ref_s_to_s_ratio)
                all_marker_kps += v_marker_kps
                all_marker_desc.append(v_marker_desc)
                h_marker_kps, h_marker_desc = self.feature_detector.detectAndCompute(h_img, None)
                for m_kps in h_marker_kps:
                    m_kps.pt = (m_kps.pt[0]*ref_s_to_s_ratio, m_kps.pt[1]*ref_s_to_s_ratio)
                all_marker_kps += h_marker_kps
                all_marker_desc.append(h_marker_desc)

        all_marker_desc = np.concatenate(all_marker_desc, axis=0)
        return all_marker_kps, all_marker_desc

    def create_motion_blurred_img(self, img, kernel_size=20):
        # Create kernels
        kernel_v = np.zeros((kernel_size, kernel_size))
        kernel_h = np.copy(kernel_v)
        kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel_v /= kernel_size
        kernel_h /= kernel_size
        # Apply the vertical kernel.
        vertical_mb = cv2.filter2D(img, -1, kernel_v)
        # Apply the horizontal kernel.
        horizonal_mb = cv2.filter2D(img, -1, kernel_h)
        return vertical_mb, horizonal_mb

    def find_marker_on_frame(self, img, show_matches=True, w_name='Matches'):
        kps, desc = self.feature_detector.detectAndCompute(img, mask=None)
        est_bbox, box_corners, matches, matches_orig = self.get_estimated_bbox(desc, kps, img)

        # debug
        if show_matches:
            img_1 = cv2.drawMatches(img, kps, self.marker_img_viz, self.marker_kps, matches, img, flags=2)
            cv2.imshow(w_name, img_1)
            img_2 = cv2.drawMatches(img, kps, self.marker_img_viz, self.marker_kps, matches_orig, img, flags=2)
            cv2.imshow(f'{w_name}_original', img_2)
        return est_bbox, box_corners

    def get_estimated_bbox(self, desc, kps, img):
        box_corners = None
        est_bbox = None
        # matches, est_obj_size, matches_orig = self.get_flann_matches(desc, kps)
        # if len(matches) < 3:
        matches, est_obj_size, matches_orig = self.get_bf_matches(desc, kps)
        if len(matches) > 3:
            valid_actual_kps_x = [kps[m.queryIdx].pt[0] for m in matches]
            valid_actual_kps_y = [kps[m.queryIdx].pt[1] for m in matches]
            est_bbox = [int(np.mean(valid_actual_kps_x)-est_obj_size/2),
                        int(np.mean(valid_actual_kps_y)-est_obj_size/2),
                        int(est_obj_size),
                        int(est_obj_size)]

            M, M_inv = self.get_affine_trf_between_ref_and_actual_kps(matches, kps, self.marker_kps)
            if M is not None:
                h, w = self.marker_img_ref_size
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                box_corners = cv2.perspectiveTransform(pts, M)
                warped_img = cv2.warpPerspective(img, M_inv, (500, 600))
                warped_marker_img = warped_img[:h, :w]
                # cv2.imshow("Backprojected Marker", warped_marker_img)
                warped_marker_img = warped_marker_img - np.min(warped_marker_img)
                warped_marker_img = warped_marker_img * (255./np.max(warped_marker_img))
                warped_marker_img = warped_marker_img.astype(np.uint8)
                white_ratio = np.mean(self.marker_img_viz)
                bin_th = np.percentile(warped_marker_img, white_ratio)

                # bin_th = np.mean(result)*1.1  # todo magic constant
                warped_marker_img[warped_marker_img <= bin_th] = 0
                warped_marker_img[warped_marker_img > bin_th] = 255
                diff_img = np.abs(self.marker_img_viz.astype(np.float32)-warped_marker_img.astype(np.float32))
                mean_diff = np.mean(diff_img)
                # print(f"Mean error of backprojection: {mean_diff}")
                # cv2.imshow("Ref Marker Img", self.marker_img_viz)
                # cv2.imshow("Backprojected Marker Bin", warped_marker_img)
                # cv2.imshow("Backprojected Error", diff_img.astype(np.uint8))
                # cv2.waitKey()
                if mean_diff > 50:  # todo magic number
                    box_corners = None
                    est_bbox = None

        return est_bbox, box_corners, matches, matches_orig

    def get_bf_matches(self, desc, kps):
        matches_orig = self.feature_matcher.match(desc, self.marker_desc)
        matches = sorted(matches_orig, key=lambda x: x.distance)
        matches, est_obj_size = self.filter_robust_matches(matches, kps)
        matches = matches[:self.nr_kps_to_use]
        return matches, est_obj_size, matches_orig

    def get_flann_matches(self, desc, kps):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches_orig = flann.knnMatch(desc, self.marker_desc, k=2)
        matches = [m for m, n in matches_orig if m.distance < 0.7 * n.distance]

        # Weaker pairing, but results more pairs
        if len(matches) < 4:
            matches_orig = self.feature_matcher.match(desc, self.marker_desc)
            matches = list(matches_orig)

        matches = sorted(matches, key=lambda x: x.distance)
        matches, est_obj_size = self.filter_robust_matches(matches, kps)
        matches = matches[:self.nr_kps_to_use]
        return matches, est_obj_size, matches_orig

    def filter_robust_matches(self, matches, kps):
        def filter_dist_ratios(dist_ratio_values, dist_ratio_matrix):
            dist_ratio_values[dist_ratio_values > 2] = np.nan  # magic number todo
            ratio_median = np.nanmedian(dist_ratio_values)
            ratio_std = np.max([0.1, mad(dist_ratio_values, axis=None, nan_policy='omit')])  # todo magic constant!!!!
            # get valid ratios: where the distance difference is within std
            valid_distance_filter = (np.abs(dist_ratio_matrix - ratio_median) / ratio_std) < 2  #2 should be a parameter todo
            return valid_distance_filter

        if len(matches) > 2:
            ref_matched_kps = [self.marker_kps[m.trainIdx] for m in matches]
            actual_matched_kps = [kps[m.queryIdx] for m in matches]
            # get reference distances between top kps
            ref_kp_distances = self.get_kp_euclidean_distance_matrix(ref_matched_kps)
            # get distances between kps on the frame
            actual_kp_distances = self.get_kp_euclidean_distance_matrix(actual_matched_kps)
            # get the ratio
            dist_ratio_matrix = actual_kp_distances / ref_kp_distances

            # get valid distances iteratively (mean and std gets better in each iteration)
            valid_distance_filter = filter_dist_ratios(np.array(dist_ratio_matrix[:]), dist_ratio_matrix)
            for i in range(3):
                valid_distance_filter = filter_dist_ratios(np.array(dist_ratio_matrix[valid_distance_filter]), dist_ratio_matrix)

            # calculate bbox size based on the calculated distance ratio
            valid_dist_ratio_matrix = dist_ratio_matrix[valid_distance_filter]
            valid_ratio_mean = np.mean(valid_dist_ratio_matrix)
            est_obj_size = self.marker_img_ref_size[0] * valid_ratio_mean

            # based on the valid and invalid distances find the good matches
            valid_matches_filter = np.ones((valid_distance_filter.shape[0]))
            min_agreement = np.min(np.nanmean(valid_distance_filter, axis=0))
            valid_distance_filter_ = np.array(valid_distance_filter)
            min_agreement_th = 0.8
            while min_agreement < min_agreement_th:
                agreement = np.nanmean(valid_distance_filter_[valid_matches_filter == 1, :], axis=0)
                agreement[valid_matches_filter == 0] = np.inf
                min_agreement = np.min(agreement)
                if min_agreement < min_agreement_th:
                    worst_match_idx = np.argmin(agreement)
                    valid_matches_filter[worst_match_idx] = 0

            # apply the filter to get the valid, robust matches
            matches = [m for i, m in enumerate(matches) if valid_matches_filter[i]]
            return matches, est_obj_size
        else:
            return matches, 50

    def get_kp_euclidean_distance_matrix(self, kps):
        nr_kps = len(kps)
        dist_array = np.empty([nr_kps, nr_kps])
        dist_array[:] = np.nan
        for idx1, kp1 in enumerate(kps):
            for idx2, kp2 in enumerate(kps):
                if idx1 != idx2:
                    dist_array[idx1, idx2] = np.sqrt((kp1.pt[0]-kp2.pt[0])**2 + (kp1.pt[1]-kp2.pt[1])**2)
        dist_array[dist_array == 0] = np.nan
        return dist_array

    def get_affine_trf_between_ref_and_actual_kps(self, matches, kps, ref_kps):
        if len(matches) > 3:
            src_pts = np.array([ref_kps[m.trainIdx].pt for m in matches]).astype(np.float32)
            dst_pts = np.array([kps[m.queryIdx].pt for m in matches]).astype(np.float32)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
            M_inv, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 10.0)
        else:
            M = None
            M_inv = None
        return M, M_inv



