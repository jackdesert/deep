from keras.datasets import mnist
import array

(train_images, _), (_, _) = mnist.load_data()

length = 27
img = train_images[1]
images = train_images[:9]

for img in images:
    for row in img:
        string = array.array('u')
        for cell in row:
            if cell > 100:
                string.append('O')
            else:
                string.append(' ')
        print(string)
