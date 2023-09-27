from PIL import Image
import numpy as np

# Open the image file
image = Image.open('rgb_ia.png')  # Replace 'your_image_file.jpg' with the path to your image file

# Convert the image to a NumPy array
image_array = np.array(image)

# Split the image into RGB channels
r, g, b = image.split()

# Convert the channels to NumPy arrays
r_array = np.array(r)
g_array = np.array(g)
b_array = np.array(b)

# Display the RGB channels as separate images
r.show(title='Red Channel')
g.show(title='Green Channel')
b.show(title='Blue Channel')

# Create arrays for the RGB channels and intensity differences
rgb_channels = np.dstack((r_array, g_array, b_array))
intensity_diff = np.max(rgb_channels, axis=2) - np.min(rgb_channels, axis=2)

# Show the intensity difference as an image
Image.fromarray(intensity_diff).show(title='Intensity Difference')

# Close the image file when you're done with it (optional but recommended)
image.close()
