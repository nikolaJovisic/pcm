import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def show(*images, filename="tmp"):
    num_images = len(images)
    if num_images == 1:
        plt.figure(figsize=(5, 5))
        plt.imshow(images[0], cmap="gray")
        plt.axis("off")
    else:
        _, axs = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
        for i, image in enumerate(images):
            axs[i].imshow(image, cmap="gray")
            axs[i].axis("off")
    plt.savefig(filename + ".png")
    plt.show()
