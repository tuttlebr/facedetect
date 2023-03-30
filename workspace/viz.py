import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec, patches

plt.style.use("dark_background")


def plot_coco_sample(
    outputs,
    relative_coords=False,
    title=None,
    dpi=60,
):
    color = "#76b900"
    images, bboxes, labels, mask_polygons, mask_vertices = outputs
    batch_size = len(outputs[0])
    fig = plt.figure(dpi=dpi)
    plt.suptitle(None)

    columns = batch_size // 2
    rows = batch_size // 2
    gs = gridspec.GridSpec(rows, columns)
    flat = 0

    for image, bbox, polygon, vertice in zip(
        images, bboxes, mask_polygons, mask_vertices
    ):
        bbox = np.array(bbox)
        polygon = np.array(polygon)
        vertice = np.array(vertice)
        H, W, C = image.shape()
        ax = plt.subplot(gs[flat])
        plt.axis("off")
        plt.title("{} Face(s)".format(len(bbox)))
        ax = plt.subplot(gs[flat])
        plt.imshow(image)

        # Bounding Boxes
        for i in range(len(bbox)):
            ax = plt.subplot(gs[flat])
            l, t, r, b = bbox[i] * [W, H, W, H] if relative_coords else bbox[i]
            rect = patches.Rectangle(
                (l, t),
                width=(r - l),
                height=(b - t),
                linewidth=1,
                edgecolor=color,
                facecolor="none",
            )

            # Segmentation Masks
            mask = polygon[:, 0] == i
            tmp_vertice = vertice[mask]

            # Scale relative coordinates to the bbox dimensions
            cx_scale, cy_scale = (r - l) / 80, (b - t) / 80
            tmp_vertice = tmp_vertice * [cx_scale, cy_scale]
            for xy in tmp_vertice:
                xy = (
                    xy[0] + l,
                    xy[1] + t,
                )
                circle = patches.Circle(xy, radius=8, color=color)
                ax.add_patch(circle)
            ax.add_patch(rect)

        flat += 1


def show(outputs, relative_coords=False, title=None, index=0, dpi=60):
    outputs = [x.as_cpu() if hasattr(x, "as_cpu") else x for x in outputs]

    plot_coco_sample(
        outputs,
        relative_coords=relative_coords,
        title=title,
        dpi=dpi,
    )
