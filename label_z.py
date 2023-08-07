import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import openpyxl
from scipy.ndimage import convolve
import numpy as np
import cv2
import os

"""README
打开图片，在图像上选择最多两个点，点击保存后打开下一张图片。
保存后会在excel文件中存入path以及两个点的横坐标，会自动将小的放前面。
但是同一张图片多次保存会保存多个数据，需要注意。

"""

class ImageAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Annotator")

        self.image_canvas = tk.Canvas(root)
        self.image_canvas.pack()

        self.open_button = tk.Button(root, text="Open Image", command=self.open_image)
        self.open_button.pack(side=tk.LEFT, padx=10)

        self.save_button = tk.Button(root, text="Save to Excel", command=self.save_to_excel)
        self.save_button.pack(side=tk.LEFT, padx=10)

        self.revo_button = tk.Button(root, text="withdraw a point", command=self.withdraw_point)
        self.revo_button.pack(side=tk.LEFT, padx=10)

        self.edge_button = tk.Button(root, text="find edge", command=self.find_edge)
        self.edge_button.pack(side=tk.LEFT, padx=10)

        self.image_path = None
        self.annotated_points = []
        self.zoom_factor = 1.0
        self.canvas_x = 0
        self.canvas_y = 0

        self.image_canvas.bind("<Button-1>", self.annotate_point)
        self.image_canvas.bind("<MouseWheel>", self.zoom)

    def withdraw_point(self):
        pt_id = self.annotated_points[-1][-1]
        self.image_canvas.delete(pt_id)
        self.annotated_points = self.annotated_points[:-1]

    def open_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
        if self.image_path:
            self.load_image()

    def load_image(self):
        self.annotated_points = []
        image = Image.open(self.image_path)
        width, height = image.size

        # Resize the Canvas to match the image size
        canvas_width = int(width * self.zoom_factor)
        canvas_height = int(height * self.zoom_factor)
        self.image_canvas.config(width=canvas_width, height=canvas_height)

        image = image.resize((canvas_width, canvas_height), Image.ANTIALIAS)

        self.photo = ImageTk.PhotoImage(image)
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def zoom(self, event):
        if self.image_path:
            if event.delta > 0:
                self.zoom_factor *= 1.2
            else:
                self.zoom_factor /= 1.2
            self.load_image()

    def annotate_point(self, event):
        if self.image_path:
            x, y = event.x, event.y
            if len(self.annotated_points) == 2:
                messagebox.showinfo("Fail", "No exceed two points")
                return
            point_id = self.image_canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="red")
            self.annotated_points.append((x, y, point_id))

            if len(self.annotated_points) == 2:
                self.draw_box()

    # def draw_box(self):
    #     x1, y1, _ = self.annotated_points[0]
    #     x2, y2, _ = self.annotated_points[1]
    #     self.image_canvas.create_rectangle(x1, y1, x2, y2, outline="blue")

    #     # Extract the region within the box
    #     box_left = min(x1, x2)
    #     box_right = max(x1, x2)
    #     box_top = min(y1, y2)
    #     box_bottom = max(y1, y2)

    #     # Get the cropped image region
    #     image = Image.open(self.image_path)
    #     cropped_image = image.crop((box_left, box_top, box_right, box_bottom))

    #     # Convert cropped image to grayscale and apply Sobel filter
    #     gray_image = cropped_image.convert("L")
    #     kernel1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    #     kernel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    #     sobel_kernel = kernel1
    #     sobel_result = convolve(np.array(gray_image), sobel_kernel)

    #     # Create an image from the Sobel result array
    #     sobel_image = Image.fromarray(sobel_result)

    #     # Display the Sobel filtered image
    #     sobel_photo = ImageTk.PhotoImage(sobel_image)
    #     self.image_canvas.create_image(self.canvas_x + box_left, self.canvas_y + box_top, anchor=tk.NW, image=sobel_photo)
    #     self.image_canvas.image = sobel_photo

    def draw_box(self):
        x1, y1, _ = self.annotated_points[0]
        x2, y2, _ = self.annotated_points[1]
        self.image_canvas.create_rectangle(x1, y1, x2, y2, outline="blue")

        # Extract the region within the box
        box_left = min(x1, x2)
        box_right = max(x1, x2)
        box_top = min(y1, y2)
        box_bottom = max(y1, y2)

        # Read the image and convert to grayscale

        self.relative_path = os.path.relpath(self.image_path, os.path.dirname(__file__))
        # print(f'path {self.relative_path}')
        image_toCrop = cv2.imread(self.relative_path, cv2.IMREAD_GRAYSCALE)
        image = image_toCrop[box_top:box_bottom, box_left:box_right]
        # print(f'image {image}')
        # Apply Canny edge detection
        edges = cv2.Canny(image, threshold1=100, threshold2=250)  # 调整阈值

        # Create a binary mask for edges above the high threshold
        high_threshold = 150  # 设定高阈值
        edges_high = (edges >= high_threshold).astype(np.uint8) * 255

        # # Create an image from the edges mask
        # edges_image = Image.fromarray(edges_high)

        # # Display the edge-detected image
        # edges_photo = ImageTk.PhotoImage(edges_image)
        # self.image_canvas.create_image(self.canvas_x + box_left, self.canvas_y + box_top, anchor=tk.NW, image=edges_photo)
        # self.image_canvas.image = edges_photo
        # self.edge_img = edges_image

        # Create an image from the edges mask
        self.edges_image = Image.fromarray(edges_high)
        self.edges_photo = ImageTk.PhotoImage(self.edges_image)

        # Display the edge-detected image
        self.image_canvas.create_image(self.canvas_x + box_left, self.canvas_y + box_top, anchor=tk.NW, image=self.edges_photo)
        self.image_canvas.image = self.edges_photo

    def find_edge(self):
        # Load the edge-detected image (you need to replace 'edges_image.png' with your image file)
        # edges_image = cv2.imread('edges_image.png', cv2.IMREAD_GRAYSCALE)
        # edges_image = np.array(self.edge_img)

        # Find contours in the edge image
        contours, _ = cv2.findContours(np.array(self.edges_image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select the first contour (you can change this index or use other criteria)
        selected_contour = contours[0]

        # Fit an ellipse to the selected contour
        ellipse = cv2.fitEllipse(selected_contour)

        # Draw the ellipse on the original image
        original_image = cv2.imread(self.relative_path)
        cv2.ellipse(original_image, ellipse, (0, 255, 0), 2)

        original_image = Image.fromarray(original_image)

        # Display the edge-detected image
        edges_photo = ImageTk.PhotoImage(original_image)
        self.image_canvas.create_image(self.canvas_x, self.canvas_y, anchor=tk.NW, image=edges_photo)
        self.image_canvas.image = edges_photo


    def save_to_excel(self):
        if self.image_path and self.annotated_points:
            excel_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
            if excel_path:
                try:
                    wb = openpyxl.load_workbook(excel_path)
                    ws = wb.active
                except FileNotFoundError:
                    wb = openpyxl.Workbook()
                    ws = wb.active
                    ws.append(["Image Path", "X1", "Y1", "X2", "Y2"])

                
                x1, y1, id = self.annotated_points[0]
                x2, y2, id = self.annotated_points[1]
                pt_min, pt_max = (x1, y1), (x2, y2)

                if x1 > x2:
                    pt_min = (x2, y2)
                    pt_max = (x1, y1)
                
                

                ws.append([self.relative_path, pt_min[0],pt_min[1], pt_max[0], pt_max[1]])

                wb.save(excel_path)
                messagebox.showinfo("Success", "Data saved to Excel file.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotator(root)
    root.mainloop()

