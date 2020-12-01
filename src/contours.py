import numpy as np
import torch
from scipy.stats import poisson
from skimage.morphology import dilation, erosion, disk


P_DEAD = 0.3381
LOC_LIVE = 4.0 
MU_LIVE = 1.0059
LOC_DEAD = 3.0
MU_DEAD = 0.3423


def sample_poisson(mu, loc, p):
    return loc + poisson(mu).ppf(p) + 1


def sample_radius():

    cls = 0 if np.random.rand() > P_DEAD else 1
 
    if cls == 0:
        r = sample_poisson(MU_LIVE + 1, LOC_LIVE, np.random.rand())
    else:
        r = sample_poisson(MU_DEAD, LOC_DEAD, np.random.rand())

    return int(r), cls


def generate_points(radii, classes, dim=128):

    ys, xs = np.nonzero(np.ones((dim, dim)))
    points = []
 
    for r, cls in zip(radii, classes):
        rand_idx = np.random.randint(ys.shape[0])
        y = ys[rand_idx]
        x = xs[rand_idx]
 
        idx = np.sqrt((ys - y) ** 2 + (xs - x) ** 2) < r
 
        ys = ys[~idx]
        xs = xs[~idx]
 
        points.append([y, x])
 
    return np.array(points)


def position_masks(points, radii, dim):

    nb_cells = len(points)

    coord_dict = {}
    sites = {}

    # read coordinates
    for k in range(1, nb_cells + 1):
        center_y, center_x = points[k - 1]
        r = radii[k - 1]
        sites[k] = (center_y, center_x, r)

        mask = disk(r)

        ys, xs = np.nonzero(mask)
        ys += center_y - r
        xs += center_x - r

        for y, x in zip(ys, xs):

            if y < 0 or x < 0 or y >= dim or x >= dim:
                continue

            if not (y, x) in coord_dict:
                coord_dict[(y, x)] = []
            coord_dict[(y, x)].append(k)
    
    return coord_dict, sites


def voronoi_masks(coord_dict, sites, dim):

    canvas = np.zeros((dim, dim))

    # assign coordinates
    for y, x in coord_dict.keys():
        regions = coord_dict[(y, x)]

        if len(regions) == 1:  # pixel only in one object
            canvas[y, x] = regions[0]

        else:  # pixel at intersection of more than one object
            min_dist = float('inf')
            for k in regions:
                center_y, center_x, r = sites[k]
                dist = (np.sqrt((center_y - y) ** 2 + (center_x - x) ** 2)) / r

                if dist < min_dist:
                    min_dist = dist
                    canvas[y, x] = k

    return canvas


def get_contours(instance_mask):
    return instance_mask - erosion(instance_mask)


def assign_contour_classes(contours, classes):

    for k in np.unique(contours)[1:]:
        cls = classes[int(k) - 1]

        if cls == 0:
            contours[contours==k] = 255.5  # not 255 in case of label clash
        else:
            contours[contours==k] = 127.5

    contours[contours==255.5] = 255

    return contours / 255.


def sample_contours(dim, min_cells, max_cells, bbox_size=None):

    nb_cells = np.random.randint(min_cells, max_cells)

    data = sorted([sample_radius() for _ in range(nb_cells)],
                  key=lambda x : x[0], reverse=True)

    radii = [r for r, _ in data]
    classes = [cls for _, cls in data]

    points = generate_points(radii, classes, dim)

    coord_dict, sites = position_masks(points, radii, dim)
    instance_mask = voronoi_masks(coord_dict, sites, dim)

    contours = get_contours(instance_mask)
    cell_contours = assign_contour_classes(contours, classes)
    cell_contours = torch.Tensor(cell_contours * 2 - 1)
 
    centers = torch.Tensor(points)

    if bbox_size is None:
        bbox_size = 2 * torch.Tensor(radii)[:, None]

    bboxes = torch.cat([centers[:, 1:] - bbox_size // 2,
                        centers[:, :1] - bbox_size // 2,
                        centers[:, 1:] + bbox_size // 2,
                        centers[:, :1] + bbox_size // 2], axis=1)

    class_idx = torch.Tensor(classes)

    return cell_contours, bboxes, class_idx


def sample_phase_contrast(imgs, all_dfs, dim, bbox_size):

    idx = np.random.randint(len(imgs))

    img = imgs[idx]
    df_features = all_dfs[idx]

    y, x = (torch.randint(dim - bbox_size, size=(1,)).item(),
            torch.randint(dim - bbox_size, size=(1,)).item())

    # extract crop and normalise
    crop = (torch.Tensor(img[y:y+dim, x:x+dim]) / 127.5) - 1

    df_slice = df_features[(df_features.xmin >= x) & (df_features.xmax < x + dim) &
                           (df_features.ymin >= y) & (df_features.ymax < y + dim)]

    coords = torch.Tensor(df_slice[['ymin', 'xmin', 'ymax', 'xmax']].values)
    y_centers = ((coords[:, 0] + coords[:, 2]) // 2 - y)[:, None]
    x_centers = ((coords[:, 1] + coords[:, 3]) // 2 - x)[:, None]

    bboxes = torch.cat([x_centers - bbox_size // 2, y_centers - bbox_size // 2,
                        x_centers + bbox_size // 2, y_centers + bbox_size // 2], axis=1)

    class_idx = torch.Tensor(df_slice['class_id'].values)

    return crop, bboxes, class_idx


def draw_conditions(bboxes, dim):

    condition = torch.zeros((1, 2, dim, dim))
    noise = torch.zeros((1, 2, dim, dim))

    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax, cls = map(int, bbox)

        ymin = max(0, ymin + 10)
        xmin = max(0, xmin + 10)

        ymax = min(dim, ymax - 10)
        xmax = min(dim, xmax - 10)

        condition[0, cls, ymin:ymax, xmin:xmax] = 1

        z = torch.randn(1, 1, ymax-ymin, xmax-xmin)
        noise[0, cls, ymin:ymax, xmin:xmax] = z

    return condition, noise
