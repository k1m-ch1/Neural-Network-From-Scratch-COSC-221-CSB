import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    images = get_images("../dataset/t10k-images.idx3-ubyte")
    labels = get_labels("../dataset/t10k-labels.idx1-ubyte")
    N = 100
    for n in range(N):
        plt.title(f"This is an image of a {labels[n]}")
        plt.imshow(images[n])
        plt.show()
