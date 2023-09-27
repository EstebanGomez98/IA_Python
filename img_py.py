from PIL import Image

# Open the image file
image = Image.open('rgb_ia.png')  # Replace 'your_image_file.jpg' with the path to your image file

# Get the size (width and height) of the image
width, height = image.size

# Print the dimensions
print(f"Image width: {width}px")
print(f"Image height: {height}px")

# You can also print them together
print(f"Image size: {width}x{height}px")

# Close the image file when you're done with it (optional but recommended)
image.close()
