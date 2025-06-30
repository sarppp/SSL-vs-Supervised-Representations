import os

def list_folder_contents(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # List all files and directories in the specified folder
        contents = os.listdir(folder_path)
        print(f"Contents of '{folder_path}':")
        for item in contents:
            print(item)
    else:
        print(f"Folder not found: '{folder_path}'")

def load_image_paths_and_labels(data_dir):
    image_paths = []
    labels = []

    # Iterate through each subfolder (class)
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)

        # Check if it's a directory
        if os.path.isdir(class_dir):
            # Iterate through each image in the class subfolder
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)

                # Check if it's a file and potentially an image file
                # You might want to add more robust image file checks
                if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(image_path)
                    labels.append(class_name)

    # Now you have a list of image paths and their corresponding labels
    # You can use these lists to load images and prepare them for further analysis or modeling
    print("First 10 image paths:", image_paths[:10])
    print("First 10 labels:", labels[:10])
    print(f"Found {len(image_paths)} images across {len(set(labels))} classes.")
    
    return image_paths, labels