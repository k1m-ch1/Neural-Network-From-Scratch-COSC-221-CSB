import numpy as np

def get_images(filename):
    with open(filename, 'rb') as file:
        magic_bytes = file.read(4)
        number_of_images = int.from_bytes(file.read(4))
        number_of_rows = int.from_bytes(file.read(4))
        number_of_columns = int.from_bytes(file.read(4))

        return [[[int.from_bytes(file.read(1)) for _ in range(number_of_columns)]
            for _ in range(number_of_rows)]
            for _ in range(number_of_images)]

def get_labels(filename):
    with open(filename, 'rb') as file:
        magic_bytes = file.read(4)
        number_of_labels = int.from_bytes(file.read(4))
        return [int.from_bytes(file.read(1)) for _ in range(number_of_labels)]

def get_images_fast(filename):
    with open(filename, 'rb') as file:
        magic_bytes = int.from_bytes(file.read(4))
        if magic_bytes != 2051:
            raise ValueError("Magic bytes doesn't match that of idx3")
        number_of_images = int.from_bytes(file.read(4))
        number_of_rows = int.from_bytes(file.read(4))
        number_of_columns = int.from_bytes(file.read(4))

        images = np.frombuffer(file.read(), dtype=np.uint8)
        images = images[:number_of_images*number_of_rows*number_of_columns]
        return images.reshape(number_of_images, number_of_rows, number_of_columns)

def get_labels_fast(filename):
    with open(filename, 'rb') as file:
        magic_bytes = int.from_bytes(file.read(4))
        if magic_bytes != 2049:
            raise ValueError("Magic bytes doesn't match that of idx1")
        number_of_labels = int.from_bytes(file.read(4))
        return np.frombuffer(file.read(), dtype=np.uint8)

if __name__ == "__main__":
    images = get_images_fast("dataset/train-images.idx3-ubyte")
    labels = get_labels_fast("dataset/train-labels.idx1-ubyte")

    #import matplotlib.pyplot as plt
    #N = 100
    #for n in range(N):
    #    plt.title(f"This is an image of a {labels[n]}")
    #    plt.imshow(images[n])
    #    print(images[n])
    #    plt.show()
