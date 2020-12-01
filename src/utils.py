import numpy as np
import torch
from torch.nn import UpsamplingNearest2d, UpsamplingBilinear2d

from rectpack import newPacker


def get_mnist_canvas(images, labels, nb_classes=10, dim=128):

    canvas = -torch.ones((dim, dim))
    noise_canvas = torch.zeros((nb_classes, dim, dim))
    condition_canvas = torch.zeros((nb_classes, dim, dim))

    num_objs, h, w = images.shape

    y, x = (torch.randint(0, dim - h, size=(num_objs, 1)),
            torch.randint(0, dim - w, size=(num_objs, 1)))

    bboxes = torch.cat([x, y, x + w, y + h], axis=1)

    for i, (x1, y1, x2, y2) in enumerate(bboxes):

        canvas[y1:y2, x1:x2] = torch.max(canvas[y1:y2, x1:x2],
                                         images[i].squeeze())

        z = torch.randn(1, 1, w // 4, h // 4)
        z = UpsamplingNearest2d(scale_factor=4)(z)

        noise_canvas[labels[i], y1:y2, x1:x2] = z.squeeze()

        condition_canvas[labels[i], y1:y2, x1:x2] = torch.ones((h, w))

    #bboxes = torch.cat([bboxes, labels[:, None]], axis=1)

    return canvas, noise_canvas, condition_canvas, bboxes


def get_mnist_knapsack(images, labels, nb_classes=10, dim=128):

    bboxes = []

    canvas = -torch.ones((dim, dim))
    noise_canvas = torch.zeros((nb_classes, dim, dim))
    condition_canvas = torch.zeros((nb_classes, dim, dim))

    hs, ws = 28 + 5 * np.random.randn(2, images.shape[0])
    hs = np.clip(hs, 14, 48).astype('int')
    ws = np.clip(ws, 14, 48).astype('int')

    rectangles = list(zip(hs, hs))
    bins = [(128, 128)]

    packer = newPacker()

    # Add the rectangles to packing queue
    for r in rectangles:
        packer.add_rect(*r)

    # Add the bins where the rectangles will be placed
    for b in bins:
        packer.add_bin(*b)

    # Start packing
    packer.pack()

    for i, rect in enumerate(packer.rect_list()):
        _, x, y, w, h, _ = rect

        scaled_crop = UpsamplingBilinear2d(size=(h, w))(images[i][None, None])
        canvas[y:y+h, x:x+w] = torch.max(canvas[y:y+h, x:x+w], scaled_crop)

        z = torch.randn(1, 1, 7, 7)
        z = UpsamplingNearest2d(size=(h, w))(z)
        noise_canvas[labels[i], y:y+h, x:x+w] = z

        condition_canvas[labels[i], y:y+h, x:x+w] = torch.ones((h, w))

        bboxes.append([x, y, x + w, y + h])

    return canvas, noise_canvas, condition_canvas, torch.Tensor(bboxes)


def mnist_canvas_generator(x_data, y_data, nb_batch, nb_obj, knapsack):

    f_canvas = get_mnist_knapsack if knapsack else get_mnist_canvas

    while True:

        batch_idx = torch.randint(x_data.shape[0], size=(nb_batch, nb_obj))
        data = [f_canvas(x_data[idx], y_data[idx]) for idx in batch_idx]

        canvas_batch = torch.cat([canvas[None, None] for canvas, _, _, _ in data])
        noise_batch = torch.cat([noise[None] for _, noise, _, _ in data])
        condition_batch = torch.cat([condition[None] for _, _, condition, _ in data])

#        bbox_batch = [torch.cat([i * torch.ones(nb_obj, 1), bboxes], axis=1)
#                  for i, (_, _, _, bboxes) in enumerate(data)]

        bbox_batch = [torch.cat([i * torch.ones(bboxes.shape[0], 1), bboxes], axis=1)
                  for i, (_, _, _, bboxes) in enumerate(data)]

        bbox_batch = torch.cat(bbox_batch, axis=0)

        yield canvas_batch, noise_batch, condition_batch, bbox_batch
