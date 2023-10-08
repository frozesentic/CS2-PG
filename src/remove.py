import os

# Specify the path to the "generated" folder
generated_folder = "generated"


def clear_generated_patterns(folder_path):
    try:
        # List all files in the "generated" folder
        files = os.listdir(folder_path)

        # Iterate through the files and remove them
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        print("All generated patterns have been cleared.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    clear_generated_patterns(generated_folder)
