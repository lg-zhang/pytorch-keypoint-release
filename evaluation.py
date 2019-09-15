import os, os.path

import numpy as np
import skimage
import cv2
import torch
import logging
import cppext

logger = logging.getLogger(__name__)


def find_points_in_mask(ax_, ay_, mask):
    ax = np.round(ax_).astype(np.int)
    ay = np.round(ay_).astype(np.int)

    valid = np.arange(ax.size)
    ok = np.logical_and.reduce(
        (ax >= 0, ax < mask.shape[1], ay >= 0, ay < mask.shape[0])
    )

    valid = valid[ok]

    valid = valid[mask[ay[ok], ax[ok]] > 0]

    in_mask = np.zeros(ax_.size, dtype=np.bool)
    in_mask[valid] = True

    return in_mask


def proj_points_apply_mask(x, y, H, mask):
    valid = np.arange(x.size)

    p = np.zeros((3, x.size), np.float32)
    p[0, :] = x.astype(np.float32)
    p[1, :] = y.astype(np.float32)
    p[2, :] = 1
    p = np.matmul(H, p)
    p[0, :] /= p[2, :]
    p[1, :] /= p[2, :]

    w = mask.shape[1]
    h = mask.shape[0]
    in_b = np.logical_and.reduce(
        (p[0, :] < w - 1, p[0, :] > 0, p[1, :] < h - 1, p[1, :] > 0)
    )
    valid = valid[in_b]

    in_mask = find_points_in_mask(p[0, in_b], p[1, in_b], mask)
    valid = valid[in_mask]

    return valid


def proj_keypts_apply_mask(keypts, H, mask):
    valid = proj_points_apply_mask(keypts[:, 0], keypts[:, 1], H, mask)
    return keypts[valid, :]


def select_keypts_by_response(keypts, top_n):
    sorted_idx = np.argsort(keypts[:, 2])[::-1]
    num = min(top_n, sorted_idx.size)
    return keypts[sorted_idx[0:num], :]


def distance_based_matching(kx1, ky1, kx2, ky2, tform, radius=5.0):
    nn_idx, nn_dist = cppext.distance_based_matching(kx1, ky1, kx2, ky2, tform)

    match_i = np.logical_and(nn_dist < radius * radius, nn_idx >= 0).nonzero()[0]
    match_j = nn_idx[match_i]

    return match_i, match_j


def prepare_image(im_path, im_rotation):
    safe_margin = 30
    im = cv2.imread(im_path, 0)

    # image mask
    im_mask = np.zeros((int(im_rotation[1]), int(im_rotation[0])))
    im_mask[safe_margin:-safe_margin, safe_margin:-safe_margin] = 1.0
    im_mask = skimage.transform.rotate(im_mask, im_rotation[2], resize=True)

    return im, im_mask


def eval_repeatability(keypts_i, keypts_j, mask_i, mask_j, H, top_n):
    keypts_i = keypts_i[find_points_in_mask(keypts_i[:, 0], keypts_i[:, 1], mask_i), :]
    keypts_j = keypts_j[find_points_in_mask(keypts_j[:, 0], keypts_j[:, 1], mask_j), :]
    if not keypts_i.shape[0] or not keypts_j.shape[0]:
        return 0, -1

    keypts_i = proj_keypts_apply_mask(keypts_i, H, mask_j)
    keypts_j = proj_keypts_apply_mask(keypts_j, np.linalg.inv(H), mask_i)
    if not keypts_i.shape[0] or not keypts_j.shape[0]:
        return 0, -1

    # sort by response
    keypts_i = select_keypts_by_response(keypts_i, top_n)
    keypts_j = select_keypts_by_response(keypts_j, top_n)

    # distance based matching
    match_i, match_j = distance_based_matching(
        keypts_i[:, 0], keypts_i[:, 1], keypts_j[:, 0], keypts_j[:, 1], H
    )

    return match_i.size, min(keypts_i.shape[0], keypts_j.shape[0])


class KeypointDetector(object):
    def __init__(self, model):
        assert model is not None
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        self.model = model.cuda()
        self.model.eval()

    def score_to_keypoints(self, score, n_selected, use_maxima):
        # blur score
        score = cv2.GaussianBlur(score, ksize=(0, 0), sigmaX=2.0)

        kx, ky = cppext.non_extrema_suppression(score, choose_maxima=use_maxima)
        kx, ky = cppext.compute_subpix_quadratic(score, kx, ky)

        # TODO: bilinear sample
        response = score[ky.astype(np.int), kx.astype(np.int)]

        idx = np.argsort(response)

        n_selected = min(n_selected, response.size)

        if use_maxima:
            idx = idx[-n_selected:]
        else:
            idx = idx[:n_selected]

        kx = kx[idx]
        ky = ky[idx]
        response = response[idx]

        return kx, ky, response

    def run_fcn(self, im_):
        h = im_.shape[2]
        w = im_.shape[3]

        data = torch.from_numpy(im_).float().cuda()
        result = self.model(data).cpu().data.numpy()[0, ...]

        if result.shape[0] == 3:
            score = result[0, ...]
        elif result.shape[0] == 2:
            score = np.sqrt(np.sum(np.power(result, 2.0), axis=0))
        else:
            score = result[0, ...]

        return score.reshape(score.shape[0], score.shape[1])

    def __call__(self, image, use_maxima, output_score_map=False):
        # reflect pad to ameliorate boundary effect
        _, _, _, pad = self.model.get_output_size(-1, -1)
        im_ = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)

        if len(im_.shape) < 3:
            im_ = np.expand_dims(im_, 2)
        # c h w
        im_ = im_.transpose((2, 0, 1))
        # n c h w
        im_ = np.expand_dims(im_, axis=0)
        im_ = im_.astype(np.float32) / 255.0

        score = self.run_fcn(im_)

        kx, ky, response = self.score_to_keypoints(score, 99999, use_maxima)

        keypts = np.zeros((kx.size, 3), dtype=np.float32)
        keypts[:, 0] = kx
        keypts[:, 1] = ky
        keypts[:, 2] = response

        if output_score_map:
            return keypts, score

        return keypts


class RepeatabilityEvalDataset(object):
    def __init__(self, folder_path):
        self.root = folder_path
        self.image_list = None
        self.homography_list = None
        self.load()

    # load the image list, homography list
    def load(self):
        with open(os.path.join(self.root, "test_imgs.txt")) as f:
            self.image_list = f.read().split()
            f.close()
        # print(self.image_list)
        self.n_pairs = len(self.image_list) // 2

        with open(os.path.join(self.root, "homography.txt")) as f:
            self.homography_list = f.read().split()
            f.close()
        # print(self.homography_list)

        self.rotation = np.genfromtxt(os.path.join(self.root, "rotation.txt"))

    # load the homography matrix
    @staticmethod
    def load_homography(p):
        with open(p) as f:
            H = np.zeros(9, np.float32)
            d = f.read().split()
            for i in range(9):
                H[i] = float(d[i])
            H = H.reshape(3, 3)
            f.close()
        return H

    # return image paths, homography matrix (from i to j)
    def get_image_pair(self, idx):
        return (
            os.path.join(self.root, self.image_list[idx]),
            self.rotation[idx, :],
            os.path.join(self.root, self.image_list[idx + self.n_pairs]),
            self.rotation[idx + self.n_pairs, :],
            self.load_homography(os.path.join(self.root, self.homography_list[idx])),
        )

    def get_num_pairs(self):
        return len(self.image_list) // 2

    def __call__(self, idx):
        return os.path.join(self.root, self.image_list[idx])

    def __len__(self):
        return len(self.image_list)


def evaluate_repeatability_on_dataset(dataset, key_det, top_n, use_maxima):
    # compute keypoints for all images
    keypts_list = []
    for idx in range(len(dataset)):
        im_path = dataset(idx)
        im_path = os.path.join(dataset.root, im_path)
        im = cv2.imread(im_path, 0)
        keypts_list.append(key_det(im, use_maxima))

    # measure repeatability
    n_matches_total = 0
    n_detected_total = 0
    for idx in range(dataset.get_num_pairs()):
        im_i_path, im_i_rot, im_j_path, im_j_rot, H = dataset.get_image_pair(idx)

        im_i, im_i_mask = prepare_image(im_i_path, im_i_rot)
        im_j, im_j_mask = prepare_image(im_j_path, im_j_rot)

        keypts_i = keypts_list[idx]
        keypts_j = keypts_list[idx + dataset.get_num_pairs()]

        n_matches, n_detected = eval_repeatability(
            keypts_i, keypts_j, im_i_mask, im_j_mask, H, top_n
        )

        n_matches_total += n_matches
        n_detected_total += n_detected

    rep = float(n_matches_total) / n_detected_total * 100.0

    return n_matches_total, n_detected_total, rep
