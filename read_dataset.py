import matplotlib.pyplot as plt
import struct

def get_images(filename):
    
    with open(filename, 'rb') as file:
        magic_bytes = file.read(4)
        number_of_images = int.from_bytes(file.read(4))
        number_of_rows = int.from_bytes(file.read(4))
        number_of_columns = int.from_bytes(file.read(4))

        #print("Number of images: ", number_of_images)
        #print("Number of rows: ", number_of_rows)
        #print("Number of columns: ", number_of_columns)

        #images = []
        #for _ in range(number_of_images):
        #    image = []
        #    for _ in range(number_of_rows):
        #        row = []
        #        for _ in range(number_of_columns):
        #            row.append(int.from_bytes(file.read(1)))
        #        image.append(row)
        #    images.append(image)
        #return images

        return [[[int.from_bytes(file.read(1)) for _ in range(number_of_columns)]
            for _ in range(number_of_rows)]
            for _ in range(number_of_images)]

if __name__ == "__main__":
    images = get_images("./dataset/t10k-images.idx3-ubyte")
    plt.imshow(images[3])
    plt.show()
