import numpy as np

def convolution(image, kernel):

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate output shape
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # Initialize output matrix
    output = np.zeros((output_height, output_width))

    # Perform convolution
    for i in range(output_height):
        for j in range(output_width):
            # Extract the region of interest from the image
            roi = image[i:i+kernel_height, j:j+kernel_width]

            # Perform element-wise multiplication between ROI and kernel,
            # and then sum them up to get a single value for the output matrix
            output[i, j] = np.sum(roi * kernel)

    return output

# Example usage:
image = np.array([[1, 2, 3, 4, 5, 6],
                  [7, 8, 9, 10, 11, 12],
                  [13, 14, 15, 16, 17, 18],
                  [19, 20, 21, 22, 23, 24],
                  [25, 26, 27, 28, 29, 30],
                  [31, 32, 33, 34, 35, 36]])

kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

result = convolution(image, kernel)
print("Convolution result:")
print(result)
