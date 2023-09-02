import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import openpyxl
import numpy as np
import cv2
import copy
from scipy.special import comb
from scipy.optimize import curve_fit
from ellipse import LsqEllipse
from PIL import Image, ImageGrab
# from utils import bezier_curve, bernstein_poly

"""
TODO:需要用智能的算法去自动寻找晶状体上表面
重构：
- 将相关的函数放在一起
- 将零散的函数放到另一个文件里
- 分成结构化
"""

class ImageAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Annotator")

        self.image_canvas = tk.Canvas(root)
        self.image_canvas.pack()

        self.frame1 = tk.Frame(root)
        self.frame1.pack()

        self.open_button = tk.Button(self.frame1, text="Open Image", command=self.open_image)
        self.open_button.pack(side=tk.LEFT, padx=10)

        self.edge_button = tk.Button(self.frame1, text="find edge", command=self.find_edge)
        self.edge_button.pack(side=tk.LEFT, padx=10)

        # Create an Entry widget for user input
        self.entry = tk.Entry(self.frame1)

        self.curv_button = tk.Button(self.frame1, text="fit curve", command=self.fit_curve)
        self.curv_button.pack(side=tk.LEFT, padx=10)

        self.oval_button = tk.Button(self.frame1, text="fit oval", command=self.find_oval2)
        self.oval_button.pack(side=tk.LEFT, padx=10)

        # 创建保存按钮
        self.save_button = tk.Button(self.frame1, text="save image", command=lambda: self.save_image)
        self.save_button.pack(side=tk.LEFT, padx=10)

        # 重画椭圆
        self.redr_button = tk.Button(self.frame1, text="redraw oval", command=lambda: self.redraw)
        self.redr_button.pack(side=tk.LEFT, padx=10)
        self.label = tk.Label(self.frame1, text="Enter deta (p0, p1, a, b, t):")
        self.label.pack(side=tk.LEFT, padx=10)
        # Create an Entry widget for user input
        self.deta = tk.Entry(self.frame1)
        self.deta.pack(side=tk.LEFT, padx=10)

        # 创建第二行frame,主要是分割睫状突，以及计算距离等工作。
        self.frame2 = tk.Frame(root)
        self.frame2.pack()
        self.test_button = tk.Button(self.frame2, text='frame2', command=self.ciliary_start)
        self.test_button.pack(side=tk.LEFT, padx=10)


        self.image_path = None
        self.annotated_points = []
        self.connect_edges = []
        self.zoom_factor = 1.0
        self.canvas_x = 0
        self.canvas_y = 0
        self.find_edge_flag = False

        self.ciliary_flag = False
        self.ciliary_draw_box = False
        self.ciliary_annotated_pts = list()

        self.image_canvas.bind("<Button-1>", self.annotate_point)
        self.image_canvas.bind("<MouseWheel>", self.zoom)

    """
    打开保存图像等操作，以及绘制标注点、放大缩小。
    """
    def open_image(self):
        self.find_edge_flag = False
        self.connect_edges = set()
        self.annotated_points = list()
        self.ciliary_flag = False
        self.ciliary_draw_box = False
        self.ciliary_annotated_pts = list()
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

    def save_image(self):
        # 获取画布上的内容
        x = self.root.winfo_rootx() + self.image_canvas.winfo_x()+85
        y = self.root.winfo_rooty() + self.image_canvas.winfo_y()+10
        x1 = x + self.image_canvas.winfo_width()+267
        y1 = y + self.image_canvas.winfo_height()+180
        image = ImageGrab.grab(bbox=(x, y, x1, y1))

        # 保存图像
        image.save("canvas_image.png")

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
            elif self.ciliary_flag:
                x, y = event.x, event.y
                print(f'point {x} {y}')
                if len(self.annotated_points) == 2:
                    messagebox.showinfo("Fail", "No exceed two points")
                    return
                point_id = self.image_canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="red")
                self.annotated_points.append((x, y, point_id))
                if len(self.annotated_points) == 2:
                    self.ciliary_draw_box = True
                    self.draw_box_ciliary()
            else:
                x, y = event.x, event.y
                print(f'point {x} {y}')
                if len(self.annotated_points) == 2:
                    messagebox.showinfo("Fail", "No exceed two points")
                    return
                point_id = self.image_canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="red")
                self.annotated_points.append((x, y, point_id))
                if len(self.annotated_points) == 2:
                    self.draw_box()

    """
    基本没用这个函数
    """
    def save_to_excel(self):
        # 将标注出的数据点保存到excel文件里，对同一名字可进行追加数据。
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

    """
    拟合前的系列操作，画出边框、边缘检测、选择边缘、拟合曲线。
    """
    def draw_box(self):
        # 画出两个点选中的方框，并对图像内进行边缘检测
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
        # 因为中文路径打不开
        image_toCrop = cv2.imdecode(np.fromfile(self.image_path,dtype=np.uint8),-1)

        image = image_toCrop[box_top:box_bottom, box_left:box_right]
        self.crop_img = copy.deepcopy(image)
        # Apply Canny edge detection
        if self.ciliary_flag:
            edges = cv2.Canny(image, threshold1=0, threshold2=20)
            high_threshold = 150
        else:
            edges = cv2.Canny(image, threshold1=100, threshold2=250)  
            high_threshold = 150  # 设定高阈值

        # Create a binary mask for edges above the high threshold
        edges_high = (edges >= high_threshold).astype(np.uint8) * 255

        # Create an image from the edges mask
        self.edges_image = Image.fromarray(edges_high)
        self.edges_photo = ImageTk.PhotoImage(self.edges_image)

        # Display the edge-detected image
        self.image_canvas.create_image(self.canvas_x + box_left, self.canvas_y + box_top, anchor=tk.NW, image=self.edges_photo)
        self.image_canvas.image = self.edges_photo
        
    def draw_box_ciliary2(self):
        # 画出两个点选中的方框，选中方框最下面的非黑色，然后将部分连接起来。
        x1, y1, _ = self.annotated_points[0]
        x2, y2, _ = self.annotated_points[1]
        self.image_canvas.create_rectangle(x1, y1, x2, y2, outline="blue")
        box_left = min(x1, x2)
        box_right = max(x1, x2)
        box_top = min(y1, y2)
        box_bottom = max(y1, y2)
        self.box_left = box_left
        self.box_top = box_top
        self.box_bottom = box_bottom
        self.box_right = box_right
        image_toCrop = cv2.imdecode(np.fromfile(self.image_path,dtype=np.uint8),-1)
        image = image_toCrop[box_top:box_bottom, box_left:box_right]
        self.crop_img = copy.deepcopy(image)

        # 选择图像中最下方。
        print(f'image type {type(image)}')
        print(f'image array {image.shape}')
        w, h, c = image.shape
        self.ciliary_edge_pts = list()
        for i in range(w):
            for j in range(h-1, 0, -1):
                if image[i, j, 0] > 10:
                    self.ciliary_edge_pts.append((i, j))
                    break
        
        self.edges_image = Image.fromarray(image)
        self.edges_photo = ImageTk.PhotoImage(self.edges_image)
        # Display the edge-detected image
        self.image_canvas.create_image(self.canvas_x + box_left, self.canvas_y + box_top, anchor=tk.NW, image=image)
        radis = 2
        for i in self.ciliary_edge_pts:
            self.image_canvas.create_oval(i[0]-radis, i[1]-radis, i[0]+radis, i[1]+radis, fill='red')
        self.image_canvas.image = self.edges_photo

    def draw_box_ciliary(self):
        # 画出两个点选中的方框，选中方框最下面的非黑色，然后将部分连接起来。
        x1, y1, _ = self.annotated_points[0]
        x2, y2, _ = self.annotated_points[1]
        self.image_canvas.create_rectangle(x1, y1, x2, y2, outline="blue")
        box_left = min(x1, x2)
        box_right = max(x1, x2)
        box_top = min(y1, y2)
        box_bottom = max(y1, y2)
        self.box_left = box_left
        self.box_top = box_top
        self.box_bottom = box_bottom
        self.box_right = box_right
        image_toCrop = cv2.imdecode(np.fromfile(self.image_path, dtype=np.uint8), -1)
        image = image_toCrop[box_top:box_bottom, box_left:box_right]
        self.crop_img = copy.deepcopy(image)

        # 选择图像中最下方。
        print(f'image type {type(image)}')
        print(f'image array {image.shape}')
        h, w, c = image.shape
        self.ciliary_edge_pts = list()
        self.ciliary_edge_pts_bottom = list()
        self.ciliary_edge_pts_right = list()
        for i in range(w):
            for j in range(h - 1, 0, -1):
                if image[j, i, 0] > 10:
                    self.ciliary_edge_pts.append((i+self.canvas_x + self.box_left, j+self.canvas_y + self.box_top))
                    self.ciliary_edge_pts_bottom.append((i+self.canvas_x + self.box_left, j+self.canvas_y + self.box_top))
                    break

        y_top = min(i[1] for i in self.ciliary_edge_pts_bottom)

        for i in range(h):
            if i + self.canvas_y + self.box_top > y_top:
                break
            for j in range(w-1, 0, -1):
                if image[i, j, 0] > 30:
                    self.ciliary_edge_pts.append((j+self.canvas_x + self.box_left, i+self.canvas_y + self.box_top))
                    self.ciliary_edge_pts_right.append((j+self.canvas_x + self.box_left, i+self.canvas_y + self.box_top))
                    break

        # 创建Image对象
        self.edges_image = Image.fromarray(image)
        self.edges_photo = ImageTk.PhotoImage(self.edges_image)

        # 显示边缘检测后的图像
        self.image_canvas.create_image(self.canvas_x + box_left, self.canvas_y + box_top, anchor=tk.NW, image=self.edges_photo)
        
        # 绘制红色点
        radius = 2
        # 注意bottom在下面，top在上面，y轴从上到下是正的
        # print(f'ciliary edge pts {self.ciliary_edge_pts}')
        # self.image_canvas.create_oval(self.canvas_x + self.box_left - radius, self.canvas_y + self.box_bottom - radius, 
        #                                 self.canvas_x + self.box_left + radius, self.canvas_y + self.box_bottom + radius, fill='red')
        # self.image_canvas.create_oval(self.canvas_x + self.box_left - radius, self.canvas_y + self.box_top - radius, 
        #                                 self.canvas_x + self.box_left + radius, self.canvas_y + self.box_top + radius, fill='yellow')
        # for i in self.ciliary_edge_pts:
        #     self.image_canvas.create_oval(i[0] - radius, i[1] - radius, 
        #                                 i[0] + radius, i[1] + radius, fill='red')

        # 在绘制红色点后，绘制连接这些点的线
        self.image_canvas.create_line(self.ciliary_edge_pts_bottom, 
                                    fill='green')
        self.image_canvas.create_line(self.ciliary_edge_pts_right, 
                                    fill='green')
         



    def find_edge(self):
        # 对检测出来的边缘提取并绘制在原图上
        self.find_edge_flag = True
        input_number = self.entry.get()
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

    def draw_choose_edge(self, x, y):
        # 标出鼠标选中的边
        # as my contours are just lines, pointPolygonTest would think they were inside the polygon
        # Reduce some efficiency by calculating the distance to determine which one is close
        near, near_value = 0, 10000000
        for i in range(len(self.contours)):
            res = cv2.pointPolygonTest(self.contours[i], (x-self.box_left, y-self.box_top), measureDist=True)
            if abs(res) < near_value:
                near = i
                near_value = abs(res)

        self.connect_edges.add(near)
        original_image = copy.deepcopy(self.crop_img)
        # 选中的加粗，其余的正常绘制
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


    """
    对选中的边进行曲线拟合
    """
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
        # TypeError: bernstein_poly() takes 3 positional arguments but 4 were given
        # The default parameter self is added
        polynomial_array = np.array([self.bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])
    
        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        bezier_out = [[[xvals[i], yvals[i]]] for i in range(len(xvals))]
        bezier_out = np.array(bezier_out, dtype=np.int32)

        return bezier_out

    def fit_curve(self):
        # 对选中的边进行曲线拟合，与上面的bezier，bernstein相联系。
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

    """
    迭代法拟合椭圆
    """
    
    def oval_model(self, x, p1, p2, a, b, t):
        # 椭圆方程，t is radian, sin use
        return ((x[0]-p1)*np.sin(t)+(x[1]-p2)*np.cos(t))**2 / a ** 2 + \
        (-(x[0]-p1)*np.cos(t)+(x[1]-p2)*np.sin(t))**2 / b ** 2 - 1
    
    def find_oval(self):
        # 迭代优化的椭圆拟合方法。
        # oval model 可以像上面这样定义吗，但至少目前的这些函数在验证的过程也是不够准的
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
        # 下面这步是想实现，只绘制在画布内的点
        conditions = (pts[:, 0] < self.image_canvas.winfo_width()) & (pts[:, 1] < self.image_canvas.winfo_height())
        filtered_pts = pts[conditions]
        self.content_in_box(filtered_pts, -1)

    def content_in_box(self, pts, choose):
        # TODO 有什么用？
        print(f'pts {pts.shape}')
        print(f'pt {pts}')
        for pt in pts:
            x = pt[0]
            y = pt[1]
            self.image_canvas.create_line(x, y, x, y, fill="green")  
        self.image_canvas.create_oval(0, 0, 5, 5, fill="red")
    
    """
    计算法获得椭圆，目前使用的，计算获得的椭圆交给上方进行进一步拟合
    """
    def find_oval2(self):
        # 利用论文的算法，直接计算得到椭圆
        x = np.array([pt[0, 0]  for pt in self.approx]) 
        y = np.array([pt[0, 1]  for pt in self.approx]) 
        x_t = list(zip(x, y))
        x_t = np.array(x_t)
        reg = LsqEllipse().fit(x_t)
        # TODO 得到的参数都是虚数，不知道为什么；但是这样没法控制参数
        p, a, b, t = reg.as_parameters()
        # print(f'oval params p {p[0]:.3f},{p[1]:.3f} a {a:.3f} b {b:.3f} t {t:.3f}')
        p, a, b, t = (abs(p[0])+30, abs(p[1])), abs(a)+150, abs(b)+170, abs(t) + 0.17
        print(f'oval params p {p[0]:.3f},{p[1]:.3f} a {a:.3f} b {b:.3f} t {t:.3f}')
        # 解耦椭圆的绘画，为得到x,y先记录参数
        self.oval_params = (p[0], p[1], a, b, t, 0)
        self.oval_angle = 0
        x, y = self.get_oval_xy()

        oval = np.array(list(zip(x, y)))
        # print(f'oval {oval.shape}')
        adjust_value_y = self.adjust_y(x_t, oval)

        self.oval_params = (p[0], p[1], a, b, t, adjust_value_y)
        
        print(f'adjust y value {adjust_value_y}')
        y -= adjust_value_y
        radis = 3
        self.oval_pts = list()
        for i in range(200):
            oval_pt = self.image_canvas.create_oval(x[i]-radis, y[i]-radis, x[i]+radis, y[i]+radis, fill="green")  
            self.oval_pts.append(oval_pt)
        
        self.oval = (x, y)
        
        # self.image_canvas.create_oval(self.box_left-5, self.box_top-5, self.box_left+5, self.box_top+5, fill="red")

    def adjust_y(self, arc, ellipse):
        # 对获得的椭圆进行微调，与图像拟合（为什么会拟合不上）
        """arc:np-num*2*1, ellpise:num*2*1 
            compute the aveg min distance
        """
        # random_indices = np.random.choice(arc.shape[0], 100, replace=False)
        n = arc.shape[0]

        min_y_distances = []

        for idx in range(n):
            # 选取第一个 ndarray 的点
            point1 = arc[idx]

            # 计算到第二个 ndarray 中所有点的 y 方向距离
            y_distance = np.min(ellipse[:, 1] - point1[1]-self.box_top)

            # 选择最小距离
            min_y_distances.append(y_distance)

        min_y_distances = np.array(min_y_distances)
        return np.mean(min_y_distances)
    
    def get_oval_xy(self):
        print(f'get oval xy')
        p0, p1, a, b, t, adjust_y = self.oval_params
        theta = np.linspace(0, 2*np.pi, 200)
        x = p0 + a * np.cos(theta) 
        y = p1 + b * np.sin(theta) 
        x_center = np.mean(x)
        y_center = np.mean(y)
        x -= x_center
        y -= y_center
        x = x * np.cos(t) - y * np.sin(t) + self.box_left + (self.box_right - self.box_left) / 2 + x_center
        y = x * np.sin(t) + y * np.cos(t) + self.box_top + (self.box_bottom - self.box_top) - 33 - adjust_y + y_center
        return x, y
    
    
    def redraw(self):
        input = self.deta.get()
        p0, p1, a, b, t = input.split()
        deta = (float(p0), float(p1), float(a), float(b), float(t))
        p0, p1, a, b, t = deta
        p0_, p1_, a_, b_, t_, adjust_y = self.oval_params
        self.oval_params = (p0_+p0, p1_+p1, a_+a,b_+b, t_, adjust_y)
        self.redraw_oval(deta)
    
    def redraw_oval(self, deta=(0, 0, 0, 0, 90)):
        print(f'redraw oval params {self.oval_params}')
        x, y = self.oval
        # deta是变化量，包括(x轴，y轴，角度)
        for i in self.oval_pts:
            self.image_canvas.delete(i)
        self.oval_pts = list()
        # reget point of oval,只调整a, b
        p0, p1, a, b, t = deta
        self.oval_angle += t
        x, y = self.get_oval_xy()
        # 角度也必须在上面改变！ 

        # x += p0
        # y += p1
        t = np.radians(self.oval_angle)
        # 这里角度的变化也会带动中心的变化，所以稍微处理一下
        x_center = np.mean(x)
        y_center = np.mean(y)
        x -= x_center
        y -= y_center
        x_f = x * np.cos(t) - y * np.sin(t)
        y_f = x * np.sin(t) + y * np.cos(t)
        x_f += x_center
        y_f += y_center

        x = x_f
        y = y_f
        self.oval = (x, y)
        radis = 2
        for i in range(200):
            oval_pt = self.image_canvas.create_oval(x[i]-radis, y[i]-radis, x[i]+radis, y[i]+radis, fill="red")  
            self.oval_pts.append(oval_pt)

    """
    计划对睫状突进行分割，另外就是距离计算【感觉这个软件设计的也真是困难】
    - 
    """
    def ciliary_start(self):
        # 主要目的还是分割出边缘，现在还是框选，之后也许用别的办法
        self.ciliary_flag = True
        self.find_edge_flag = False
        self.annotated_points = list()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotator(root)
    root.mainloop()

