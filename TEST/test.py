from PIL import Image
import os

# Get the current working directory (where the script is located)
current_directory = os.getcwd()

# Ask the user for the path to the input image
input_image_path = input("Enter the path to the input image: ")

# Construct the full file path for the input image
input_path = os.path.join(current_directory, input_image_path)

# Verify if the input file exists
if not os.path.exists(input_path):
    print("Error: The specified file does not exist.")
else:
    # Specify the output directory
    output_directory = os.path.join(current_directory, "output_images")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Open the image using Pillow
    img = Image.open(input_path)

    # Resize the image to 24x24 pixels
    img_resized = img.resize((24, 24))

    # Construct the full file path for the output image
    output_file_name = os.path.basename(input_path)
    output_path = os.path.join(output_directory, output_file_name)

    # Save the resized image
    img_resized.save(output_path)

    print(
        f"Resized {output_file_name} to 24x24 pixels and saved to {output_path}")
