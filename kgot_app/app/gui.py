import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Button, Label, Progressbar, Scrollbar, Frame
import os
from PIL import Image, ImageTk  # For image preview
from application import circulate_image_from_coco


def select_json():
    global json_file_path
    json_file_path = filedialog.askopenfilename(title="Select COCO JSON File", filetypes=[("JSON Files", "*.json")])
    if json_file_path:
        json_label.config(text=f"Selected: {os.path.basename(json_file_path)}")


def select_image_folder():
    global image_folder
    image_folder = filedialog.askdirectory(title="Select Image Folder")
    if image_folder:
        image_folder_label.config(text=f"Selected: {image_folder}")
        load_image_preview(image_folder)  # Load preview images when folder is selected


def select_output_folder():
    global output_folder
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if output_folder:
        output_folder_label.config(text=f"Selected: {output_folder}")


def load_image_preview(folder):
    """
    Displays the first .jpg or .png image in the specified folder as a preview.
    """
    global preview_label, preview_image

    # Find the first .jpg or .png image in the folder
    for file in os.listdir(folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(folder, file)
            try:
                img = Image.open(image_path)
                img.thumbnail((400, 400))  # Resize for preview
                preview_image = ImageTk.PhotoImage(img)
                preview_label.config(image=preview_image, text="")
                preview_label.image = preview_image
                return
            except Exception as e:
                print(f"Error loading image {file}: {e}")

    # If no valid image is found, clear the preview
    preview_label.config(image="", text="No preview available")


def run_processing():
    if not json_file_path or not image_folder or not output_folder:
        messagebox.showerror("Error", "Please select all required paths!")
        return

    try:
        # Get all image files in the folder
        image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")]
        total_images = len(image_files)

        if total_images == 0:
            messagebox.showerror("Error", "No valid images found in the selected folder!")
            return

        # Reset progress bar
        progress_bar["value"] = 0
        progress_bar["maximum"] = total_images

        # Run the processing function
        for i in range(total_images):
            # Simulating processing each image (replace with actual processing function)
            circulate_image_from_coco(
                json_file_path,
                image_folder,
                output_folder,
                output_width=1024,
                output_height=1024,
                scale=True,
            )
            # Update progress bar
            progress_bar["value"] = i + 1
            root.update_idletasks()  # Update the GUI to reflect changes in real-time

        # Display success message
        messagebox.showinfo("Success", f"Processed {total_images} images with KGOT augmentation successfully!")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def on_mouse_wheel(event):
    """
    Scroll the canvas when the mouse wheel is used.
    """
    canvas.yview_scroll(-1 * (event.delta // 120), "units")


# GUI Setup
root = tk.Tk()
root.title("KGOT Image Processor")
root.geometry("500x600")  # Default window size

# Create a scrollable frame
canvas = tk.Canvas(root)
scrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = Frame(canvas)

# Configure the canvas and scrollbar
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Bind mouse wheel scrolling to the canvas
canvas.bind_all("<MouseWheel>", on_mouse_wheel)  # For Windows and MacOS
canvas.bind_all("<Button-4>", on_mouse_wheel)   # For Linux (scroll up)
canvas.bind_all("<Button-5>", on_mouse_wheel)   # For Linux (scroll down)

# Layout the canvas and scrollbar
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Add widgets to the scrollable frame
description = (
    "Welcome to the KGOT Image Processor!\n\n"
    "1. Select the image folder containing .jpg or .png images.\n"
    "2. Select the COCO JSON file for annotations.\n"
    "3. Select the output folder to save processed images.\n\n"
    "The program will process the images using the KGOT pipeline and export them to the output folder."
)
description_label = Label(scrollable_frame, text=description, wraplength=450, justify="left")
description_label.pack(pady=10)

# JSON Selection
Button(scrollable_frame, text="Select COCO JSON", command=select_json).pack(pady=5)
json_label = Label(scrollable_frame, text="No JSON selected")
json_label.pack()

# Image Folder Selection
Button(scrollable_frame, text="Select Image Folder", command=select_image_folder).pack(pady=5)
image_folder_label = Label(scrollable_frame, text="No folder selected")
image_folder_label.pack()

# Image Preview
preview_label = Label(scrollable_frame, text="No preview available", compound="top")
preview_label.pack(pady=10)

# Output Folder Selection
Button(scrollable_frame, text="Select Output Folder", command=select_output_folder).pack(pady=5)
output_folder_label = Label(scrollable_frame, text="No folder selected")
output_folder_label.pack()

# Progress Bar
progress_bar = Progressbar(scrollable_frame, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=10)

# Run Button
Button(scrollable_frame, text="Run Processing", command=run_processing).pack(pady=20)

# Run the Tkinter main loop
root.mainloop()