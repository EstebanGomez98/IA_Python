from PIL import Image
import numpy as np

# Open the image file
# Replace 'your_image_file.jpg' with the path to your image file
image = Image.open('rgb_ia.png')

# Convert the image to a NumPy array
image_array = np.array(image)

# Split the image into RGB channels
r, g, b = image.split()

# Binarize each channel
threshold = 128  # Adjust this threshold as needed
r_bin = np.where(np.array(r) > threshold, 255, 0).astype(np.uint8)
g_bin = np.where(np.array(g) > threshold, 255, 0).astype(np.uint8)
b_bin = np.where(np.array(b) > threshold, 255, 0).astype(np.uint8)

# Convert the binary arrays back to PIL images
r_bin_image = Image.fromarray(r_bin)
# g_bin_image = Image.fromarray(g_bin)
# b_bin_image = Image.fromarray(b_bin)

print("R: ", r_bin)


# Show each binarized RGB channel as a separate image
r_bin_image.show(title='Binarized Red Channel')
# g_bin_image.show(title='Binarized Green Channel')
# b_bin_image.show(title='Binarized Blue Channel')

# Close the image files when you're done with them (optional but recommended)
image.close()
