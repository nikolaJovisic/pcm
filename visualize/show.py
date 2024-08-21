import matplotlib.pyplot as plt

def show(*images):
    num_images = len(images)
    _, axs = plt.subplots(1, num_images, figsize=(5 * num_images, 5)) 
    for i, image in enumerate(images):
        axs[i].imshow(image, cmap='gray')
        axs[i].axis('off')
    plt.show()
