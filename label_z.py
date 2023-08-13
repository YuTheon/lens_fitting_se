import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import openpyxl
import numpy as np
import cv2
import os
import copy
from scipy.special import comb
from scipy.optimize import curve_fit
from ellipse import LsqEllipse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
# from utils import bezier_curve, bernstein_poly

"""README
打开图片，在图像上选择最多两个点，点击保存后打开下一张图片。
保存后会在excel文件中存入path以及两个点的横坐标，会自动将小的放前面。
但是同一张图片多次保存会保存多个数据，需要注意。

"""
"""
TODO:需要用智能的算法去自动寻找晶状体上表面
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

        self.box_button = tk.Button(root, text="draw box", command=self.draw_box)
        self.box_button.pack(side=tk.LEFT, padx=10)

        self.edge_button = tk.Button(root, text="find edge", command=self.find_edge)
        self.edge_button.pack(side=tk.LEFT, padx=10)

        # Create a label
        self.label = tk.Label(root, text="Enter a number:")
        self.label.pack(side=tk.LEFT, padx=10)

        # Create an Entry widget for user input
        self.entry = tk.Entry(root)
        self.entry.pack(side=tk.LEFT, padx=10)

        self.curv_button = tk.Button(root, text="fit curve", command=self.fit_curve)
        self.curv_button.pack(side=tk.LEFT, padx=10)

        self.oval_button = tk.Button(root, text="fit oval", command=self.find_oval2)
        self.oval_button.pack()

        self.image_path = None
        self.annotated_points = []
        self.connect_edges = []
        self.zoom_factor = 1.0
        self.canvas_x = 0
        self.canvas_y = 0
        self.find_edge_flag = False

        self.image_canvas.bind("<Button-1>", self.annotate_point)
        self.image_canvas.bind("<MouseWheel>", self.zoom)

    def withdraw_point(self):
        pt_id = self.annotated_points[-1][-1]
        self.image_canvas.delete(pt_id)
        self.annotated_points = self.annotated_points[:-1]

    def open_image(self):
        self.find_edge_flag = False
        self.connect_edges = set()
        self.annotated_points = list()
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
        if self.image_path:
            self.load_image()

    def load_image(self):
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
            if self.find_edge_flag:
                x, y = event.x, event.y
                print(f'point {x} {y}')
                # 找到点在框框内，首先展示所有的边，点击一次后最近的边会变粗，最后用选中的曲线拟合椭圆（并且要用条件去限制）   
                self.draw_choose_edge(x, y)   
                point_id = self.image_canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="green")  
                
            else:
                x, y = event.x, event.y
                if len(self.annotated_points) == 2:
                    messagebox.showinfo("Fail", "No exceed two points")
                    return
                point_id = self.image_canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="red")
                self.annotated_points.append((x, y, point_id))

                if len(self.annotated_points) == 2:
                    self.draw_box()

    def draw_choose_edge(self, x, y):
        print(f'pt {x} {y}, box {self.box_left} {self.box_top}')
        # as my contours are just lines, pointPolygonTest would think they were inside the polygon
        # Reduce some efficiency by calculating the distance to determine which one is close
        near, near_value = 0, 10000000
        for i in range(len(self.contours)):
            res = cv2.pointPolygonTest(self.contours[i], (x-self.box_left, y-self.box_top), measureDist=True)
            if abs(res) < near_value:
                near = i
                near_value = abs(res)
            # print(f'contour {i} = {res} near {near}')

        self.connect_edges.add(near)
        original_image = copy.deepcopy(self.crop_img)

        for i in range(len(self.contours)):
            if i in self.connect_edges:
                cv2.drawContours(original_image, self.contours, i, (255, 200, 0), 2)
            else:
                cv2.drawContours(original_image, self.contours, i, (255, 0, 0), 1)

        original_image = Image.fromarray(original_image)
        # Display the edge-detected image
        edges_photo = ImageTk.PhotoImage(original_image)
        self.image_canvas.create_image(self.canvas_x+self.box_left, self.canvas_y+self.box_top, anchor=tk.NW, image=edges_photo)
        self.image_canvas.image = edges_photo


    def draw_box(self):
        x1, y1, _ = self.annotated_points[0]
        x2, y2, _ = self.annotated_points[1]
        self.image_canvas.create_rectangle(x1, y1, x2, y2, outline="blue")

        # Extract the region within the box
        box_left = min(x1, x2)
        box_right = max(x1, x2)
        box_top = min(y1, y2)
        box_bottom = max(y1, y2)
        self.box_left = box_left
        self.box_top = box_top
        self.box_bottom = box_bottom
        self.box_right = box_right

        # Read the image and convert to grayscale
        self.relative_path = os.path.relpath(self.image_path, os.path.dirname(__file__))
        image_toCrop = cv2.imread(self.relative_path)
        image = image_toCrop[box_top:box_bottom, box_left:box_right]
        self.crop_img = copy.deepcopy(image)
        # Apply Canny edge detection
        edges = cv2.Canny(image, threshold1=100, threshold2=250)  # 调整阈值

        # Create a binary mask for edges above the high threshold
        high_threshold = 150  # 设定高阈值
        edges_high = (edges >= high_threshold).astype(np.uint8) * 255

        # Create an image from the edges mask
        self.edges_image = Image.fromarray(edges_high)
        self.edges_photo = ImageTk.PhotoImage(self.edges_image)

        # Display the edge-detected image
        self.image_canvas.create_image(self.canvas_x + box_left, self.canvas_y + box_top, anchor=tk.NW, image=self.edges_photo)
        self.image_canvas.image = self.edges_photo

    def find_edge(self):
        self.find_edge_flag = True
        input_number = self.entry.get()
        # print(f'type {type(input_number)} value {input_number}')
       # Find contours in the edge image
        self.contours, _ = cv2.findContours(np.array(self.edges_image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select the first contour (you can change this index or use other criteria)
        if input_number == "":
            input_number = -1
        else:
            input_number = int(input_number)
        if input_number >= len(self.contours):
            input_number = len(self.contours)-1
        if input_number == -1:
            selected_contour = self.contours[:]
        else:
            selected_contour = self.contours[input_number]
        
        # print(f'selected contours {type(selected_contour)} {selected_contour}')

        # Fit an ellipse to the selected contour
        # ellipse = cv2.fitEllipse(selected_contour)

        # Draw the ellipse on the original image
        original_image = copy.deepcopy(self.crop_img)
        cv2.drawContours(original_image, self.contours, input_number, (255, 0, 0), 1)
        # cv2.ellipse(original_image, ellipse, (0, 0, 0), 2)

        original_image = Image.fromarray(original_image)

        # Display the edge-detected image
        edges_photo = ImageTk.PhotoImage(original_image)
        self.image_canvas.create_image(self.canvas_x+self.box_left, self.canvas_y+self.box_top, anchor=tk.NW, image=edges_photo)
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

    def bernstein_poly(self, i, n, t):
        """
        The Bernstein polynomial of n, i as a function of t
        """
        return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    def bezier_curve(self, points, nTimes=1000):
        """
        Given a set of control points, return the
        bezier curve defined by the control points.

        points should be a matrix of numpy
        such as [ [1,1], 
                    [2,3], 
                    [4,5], ..[Xn, Yn] ]-
            nTimes is the number of time steps, defaults to 1000
            See http://processingjs.nihongoresources.com/bezierinfo/
        """

        nPoints = len(points)
        xPoints, yPoints = points[:,0], points[:,1]
        bezier_out = []
        t = np.linspace(0.0, 1.0, nTimes)
        # print(f'bernstein {(0, nPoints-1, t)}')
        # TypeError: bernstein_poly() takes 3 positional arguments but 4 were given
        # The default parameter self is added
        polynomial_array = np.array([self.bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])
    
        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        bezier_out = [[[xvals[i], yvals[i]]] for i in range(len(xvals))]
        bezier_out = np.array(bezier_out, dtype=np.int32)

        return bezier_out

    def fit_curve(self):

        """
        Add the selected curves and then try to fit the curves
        """
        selected_curve = None
        for cont in self.connect_edges:
            if selected_curve is None:
                selected_curve = self.contours[cont]
            else:
                selected_curve = np.concatenate((selected_curve, self.contours[cont]), axis=0)
        cont0 = selected_curve.squeeze()
        self.approx = self.bezier_curve(cont0, nTimes=100)

        original_image = copy.deepcopy(self.crop_img)
        cv2.drawContours(original_image, self.approx, -1, (0, 255, 0), 2)
        original_image = Image.fromarray(original_image)
        # Display the edge-detected image
        edges_photo = ImageTk.PhotoImage(original_image)
        self.image_canvas.create_image(self.canvas_x+self.box_left, self.canvas_y+self.box_top, anchor=tk.NW, image=edges_photo)
        self.image_canvas.image = edges_photo
    
    def oval_model(self, x, p1, p2, a, b, t):
        """ t is radian, sin use"""
        # return ((x[0]-p1)*np.cos(t)+(x[1]-p2)*np.sin(t))**2 / a ** 2 + \
        # (-(x[0]-p1)*np.sin(t)+(x[1]-p2)*np.cos(t))**2 / b ** 2 - 1
        return ((x[0]-p1)*np.sin(t)+(x[1]-p2)*np.cos(t))**2 / a ** 2 + \
        (-(x[0]-p1)*np.cos(t)+(x[1]-p2)*np.sin(t))**2 / b ** 2 - 1
    
    def find_oval(self):
        """oval model 可以像上面这样定义吗，但至少目前的这些函数在验证的过程也是不够准的"""
        print(f'find oval {self.approx}')
        print(f'find oval type {type(self.approx)}')
        print(f'find oval shape {self.approx.shape}')
        x = np.array([pt[0, 0]  for pt in self.approx])
        y = np.array([pt[0, 1]  for pt in self.approx])
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        curve_pts = list(zip(x, y))
        print(f'curve {curve_pts}')

        init_guess = (x_max, y_max, x_max-x_min, (y_max-y_min)*2, 0)
        params_bounds = ([x_min, y_max, x_max - x_min, 1*(y_max-y_min),0], [x_max+200, y_max+400, 3*(x_max-x_min), 10*(x_max- x_min), np.pi / 2])
        self.params, _ = curve_fit(self.oval_model, x, y, p0=init_guess, bounds=params_bounds)
        print(f'find the params {self.params} var {_}')

        # draw the ellipse
        # 生成角度值
        theta = np.linspace(0, 2*np.pi, 100)

        # 计算椭圆上每个点的坐标
        p0, p1 = self.params[0], self.params[1]
        a, b, t = self.params[2], self.params[3], self.params[4] #+ np.pi/2
        # x = p0 + a * np.cos(theta) * np.sin(t) + b * np.sin(theta) * np.cos(t) + self.box_left
        # y = p1 + a * np.cos(theta) * np.cos(t) + b * np.sin(theta) * np.sin(t) + self.box_top
        x = p0 + a * np.cos(theta) * np.cos(t) + b * np.sin(theta) * np.sin(t) + self.box_left #- 120
        y = p1 - a * np.cos(theta) * np.sin(t) - b * np.sin(theta) * np.cos(t) + self.box_bottom
        # x = (a*np.cos(t)-b*np.sin(t)+p0*np.cos(theta)-np.sin(theta)*np.tan(theta)*p0) / (np.cos(theta) - np.sin(theta) * np.tan(theta))
        # y = (b*np.sin(t) - (x - p0)*np.sin(theta)) / np.cos(theta) + p1
        pts = np.array(list(zip(x, y)))
        conditions = (pts[:, 0] < self.image_canvas.winfo_width()) & (pts[:, 1] < self.image_canvas.winfo_height())
        filtered_pts = pts[conditions]
        self.content_in_box(filtered_pts, -1)

    def content_in_box(self, pts, choose):
        # filter pt that out of the figure
        # original_image = copy.deepcopy(self.crop_img)
        # cv2.drawContours(original_image, pts, choose, (0, 255, 0), 2)
        # original_image = Image.fromarray(original_image)
        # # Display the edge-detected image
        # edges_photo = ImageTk.PhotoImage(original_image)
        # self.image_canvas.create_image(self.canvas_x+self.box_left, self.canvas_y+self.box_top, anchor=tk.NW, image=edges_photo)
        # self.image_canvas.image = edges_photo
        print(f'pts {pts.shape}')
        print(f'pt {pts}')
        for pt in pts:
            x = pt[0]
            y = pt[1]
            self.image_canvas.create_line(x, y, x, y, fill="green")  
        self.image_canvas.create_oval(0, 0, 5, 5, fill="red")
    
    def find_oval2(self):
        x = np.array([pt[0, 0]  for pt in self.approx]) 
        y = np.array([pt[0, 1]  for pt in self.approx]) 
        x_t = list(zip(x, y))
        x_s = [str(i) for i in x_t]
        x_t_s = ";".join(x_s)
        print(f'sample pts {x_t_s}')
        x_t = np.array(x_t)
        reg = LsqEllipse().fit(x_t)
        # TODO 得到的参数都是虚数，不知道为什么；但是这样没法控制参数
        p, a, b, t = reg.as_parameters()
        print(f'oval params p {p[0]:.3f},{p[1]:.3f} a {a:.3f} b {b:.3f} t {t:.3f}')
        p, a, b, t = (abs(p[0]), abs(p[1])), abs(a), abs(b), abs(t)
        print(f'oval params p {p[0]:.3f},{p[1]:.3f} a {a:.3f} b {b:.3f} t {t:.3f}')

        theta = np.linspace(0, 2*np.pi, 200)
        x = p[0] + a * np.cos(theta) 
        y = p[1] + b * np.sin(theta) 
        x = x * np.cos(t) - y * np.sin(t) + self.box_left + (self.box_right - self.box_left) / 2
        y = x * np.sin(t) + y * np.cos(t) + self.box_top #+ (self.box_bottom - self.box_top) / 2

        x_t = np.array(list(zip(x, y)))
        # print(f'x, y {x_t}')
        
        for i in range(200):
            self.image_canvas.create_oval(x[i]-3, y[i]-3, x[i]+3, y[i]+3, fill="green")  
            # self.image_canvas.create_oval(x[i] - 3, y[i] - 3, x[i] + 3, y[i] + 3, fill="green")  
        
        self.image_canvas.create_oval(self.box_left-5, self.box_top-5, self.box_left+5, self.box_top+5, fill="red")



if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotator(root)
    root.mainloop()

