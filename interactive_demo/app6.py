# --- START OF FILE app6.py ---

import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import threading
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from interactive_demo.canvas2 import CanvasImage
from interactive_demo.controller2 import InteractiveController
# from interactive_demo.controller import InteractiveController
from interactive_demo.wrappers import BoundedNumericalEntry, FocusHorizontalScale, FocusCheckButton, \
    FocusButton, FocusLabelFrame


class InteractiveDemoApp(ttk.Frame):
    def __init__(self, master, args, c2t_model, matting_model):
        super().__init__(master)
        self.master = master
        master.title("交互式视频抠图演示系统")
        master.withdraw()
        master.update_idletasks()
        
        # 初始窗口大小与居中
        master.geometry("1400x750")
        x = (master.winfo_screenwidth() - 1300) / 2
        y = (master.winfo_screenheight() - 850) / 2
        master.geometry("+%d+%d" % (x, y))
        self.pack(fill="both", expand=True)

        self.limit_longest_size = args.limit_longest_size
        self.c2t_model = c2t_model
        self.args = args

        self.controller = InteractiveController(c2t_model, args.device,
                                                predictor_params={'brs_mode': 'NoBRS'},
                                                update_image_callback=self._update_image)
        # self.controller.matting_model = matting_model

        # 全局样式配置
        self._apply_global_style()
        self._init_state()
        self._add_menu()
        self._build_layout()

        master.bind('<space>', lambda event: self.controller.finish_object())
        master.bind('a', lambda event: self.controller.partially_finish_object())

        # 保持后台逻辑追踪
        self.state['zoomin_params']['skip_clicks'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['target_size'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['expansion_ratio'].trace(mode='w', callback=self._reset_predictor)
        self._reset_predictor()

        # 视频相关状态
        self.video_capture = None  # cv2.VideoCapture 对象
        self.video_path = None  # 当前视频路径
        self.video_frames = []  # 视频帧列表
        self.current_frame_index = 0  # 当前帧索引
        self.is_playing = False  # 播放状态
        self.video_play_thread = None  # 视频播放线程
        self.frames_folder = None  # 视频拆帧保存文件夹
        self.video_photo = None  # 视频预览区域的 PhotoImage 对象
        self.result_photo = None  # 结果预览区域的 PhotoImage 对象
        self.progress_window = None  # 进度条窗口
        self.video_fps = 30  # 视频帧率
        
        self._is_scribbling = False
        self._prev_scribble_x = None
        self._prev_scribble_y = None

    def _apply_global_style(self):
        """配置全局现代化样式"""
        self.bg_main = '#F0F2F5'
        self.bg_panel = '#FFFFFF'
        self.bg_canvas = '#1E1E1E'
        self.color_text = '#333333'

        self.font_title = ("Microsoft YaHei", 11, "bold")
        self.font_normal = ("Microsoft YaHei", 10)

        # 使用 ttk.Style 配置样式
        style = ttk.Style(self.master)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        style.configure('TFrame', background=self.bg_main)
        style.configure('.', font=self.font_normal, background=self.bg_panel, foreground=self.color_text)
        style.configure('TRadiobutton', background=self.bg_panel, font=self.font_normal)

        # 设置主窗口背景色
        self.master.configure(bg=self.bg_main)
    def _style_button(self, btn, btn_type='default'):
        """颜色块按钮与悬浮(Hover)动画"""
        if btn_type == 'primary':    # 蓝色 (播放/加载等主要行动)
            normal_bg, hover_bg, fg = '#1890FF', '#40A9FF', '#FFFFFF'
            font = ("Microsoft YaHei", 10, "bold")
        elif btn_type == 'success':  # 绿色 (开始抠图)
            normal_bg, hover_bg, fg = '#52C41A', '#73D13D', '#FFFFFF'
            font = ("Microsoft YaHei", 11, "bold")
        elif btn_type == 'danger':   # 红色 (重置/退出)
            normal_bg, hover_bg, fg = '#FF4D4F', '#FF7875', '#FFFFFF'
            font = ("Microsoft YaHei", 10, "bold")
        elif btn_type == 'warning':  # 橙色 (暂停/撤销)
            normal_bg, hover_bg, fg = '#FAAD14', '#FFC53D', '#FFFFFF'
            font = ("Microsoft YaHei", 10, "bold")
        elif btn_type == 'navbar':   # 顶部导航白底按钮
            normal_bg, hover_bg, fg = self.bg_panel, '#F0F2F5', self.color_text
            font = self.font_normal
        else:                        # 默认
            normal_bg, hover_bg, fg = '#E2E2E2', '#D5D5D5', self.color_text
            font = self.font_normal

        try:
            btn.configure(bg=normal_bg, fg=fg, font=font, relief=tk.FLAT, bd=0,
                          activebackground=hover_bg, activeforeground=fg, cursor="hand2")
        except:
            pass

        # 悬浮事件绑定
        def on_enter(e):
            if btn['state'] == tk.NORMAL: btn.configure(bg=hover_bg)
        def on_leave(e):
            if btn['state'] == tk.NORMAL: btn.configure(bg=normal_bg)
            
        btn.bind("<Enter>", on_enter, add='+')
        btn.bind("<Leave>", on_leave, add='+')

    def _init_state(self):
        self.state = {
            'interact_type': tk.StringVar(value='click'), 
            'scribble_area': tk.StringVar(value='foreground'), # 涂鸦区域选择: foreground/background/unknown
            'alpha_blend': tk.DoubleVar(value=0.5),       
            'click_radius': tk.IntVar(value=3),           
            'zoomin_params': {
                'use_zoom_in': tk.BooleanVar(value=True),
                'fixed_crop': tk.BooleanVar(value=True),
                'skip_clicks': tk.IntVar(value=-1),
                'target_size': tk.IntVar(value=448),
                'expansion_ratio': tk.DoubleVar(value=1.4)
            },
            'predictor_params': {'net_clicks_limit': tk.IntVar(value=8)},
            'brs_mode': tk.StringVar(value='NoBRS'),
            'prob_thresh': tk.DoubleVar(value=0.5),
            'lbfgs_max_iters': tk.IntVar(value=20),
        }

    def _add_menu(self):
        self.menubar = tk.Frame(self, bg=self.bg_panel, height=50)
        self.menubar.pack(side=tk.TOP, fill='x')
        tk.Frame(self, bg='#DCDDE1', height=1).pack(side=tk.TOP, fill='x')

        btn_container = tk.Frame(self.menubar, bg=self.bg_panel)
        btn_container.pack(side=tk.LEFT, padx=10, pady=5)

        btn_load = FocusButton(btn_container, text=' 加载文件 ', command=self._load_file_callback)
        btn_load.pack(side=tk.LEFT, padx=5, ipady=3)
        self._style_button(btn_load, 'primary')

        self.save_mask_btn = FocusButton(btn_container, text=' 保存三分图 ', command=self._save_mask_callback)
        self.save_mask_btn.pack(side=tk.LEFT, padx=5, ipady=3)
        self.save_mask_btn.configure(state=tk.DISABLED)
        self._style_button(self.save_mask_btn, 'navbar')

        self.save_video_btn = FocusButton(btn_container, text=' 保存结果视频 ', command=self._save_video_callback)
        self.save_video_btn.pack(side=tk.LEFT, padx=5, ipady=3)
        self.save_video_btn.configure(state=tk.DISABLED)
        self._style_button(self.save_video_btn, 'navbar')

        btn_exit = FocusButton(self.menubar, text=' 退出 ', command=self.master.quit)
        btn_exit.pack(side=tk.RIGHT, padx=15, pady=5, ipady=3)
        self._style_button(btn_exit, 'danger')

    def _build_layout(self):
        self.main_frame = tk.Frame(self, bg=self.bg_main)
        self.main_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        self.main_frame.rowconfigure(0, weight=1, minsize=270, uniform='row')
        self.main_frame.rowconfigure(1, weight=1, minsize=270, uniform='row')
        self.main_frame.columnconfigure(0, weight=1, minsize=480, uniform='canvas_col') #图像画布/视频预览 minsize最小宽度
        self.main_frame.columnconfigure(1, weight=1, minsize=480, uniform='canvas_col') #三分图画布/结果预览 
        self.main_frame.columnconfigure(2, weight=0, minsize=340)

        # 第 1 行
        self._add_canvas_img(0, 0)
        self._add_canvas_trimap(0, 1)
        self._add_interact_controls(0, 2)

        # 第 2 行
        self._add_canvas_video(1, 0)
        self._add_canvas_result(1, 1)
        self._add_video_controls(1, 2)

    def _create_styled_canvas_frame(self, r, c, title, placeholder_text):
        """生成统一风格的暗色画布容器"""
        # 给 FocusLabelFrame 传递 bg 属性彻底消灭灰色背景
        frame = FocusLabelFrame(self.main_frame, text=title, bg=self.bg_panel, font=self.font_title, fg=self.color_text)
        frame.grid(row=r, column=c, sticky='nswe', padx=8, pady=8)
        
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        canvas = tk.Canvas(frame, bg=self.bg_canvas, highlightthickness=0, cursor="crosshair")
        canvas.grid(row=0, column=0, sticky='nswe', padx=4, pady=4)
        canvas.bind("<Configure>", lambda e, c=canvas, t=placeholder_text: self._update_placeholder(e, c, t))
        
        return frame, canvas

    def _update_placeholder(self, event, canvas, text):
        canvas.delete("placeholder")
        canvas.create_text(event.width/2, event.height/2, text=text, 
                           fill="#666666", font=("Microsoft YaHei", 12), tags="placeholder")

    def _add_canvas_img(self, r, c):
        self.canvas_frame, self.canvas = self._create_styled_canvas_frame(r, c, " 图像画布 ", "等待加载图像...")
        self.image_on_canvas = None

    def _add_canvas_trimap(self, r, c):
        self.canvas_frame_trimap, self.canvas_trimap = self._create_styled_canvas_frame(r, c, " 三分图画布 ", "等待生成三分图...")
        self.trimap_on_canvas = None

    def _create_radio_button(self, parent, text, variable, value):
        """创建原生透明白底的单选按钮，杜绝灰色背景"""
        rb = tk.Radiobutton(parent, text=text, variable=variable, value=value, 
                            bg=self.bg_panel, activebackground=self.bg_panel, 
                            selectcolor=self.bg_panel, font=self.font_normal, fg=self.color_text)
        return rb

    def _add_interact_controls(self, r, c):
        self.interact_frame = FocusLabelFrame(self.main_frame, text=" 交互控制栏 ", bg=self.bg_panel, font=self.font_title, fg=self.color_text)
        self.interact_frame.grid(row=r, column=c, sticky='nswe', padx=8, pady=8)

        # 1. 交互类型切换
        type_frame = tk.Frame(self.interact_frame, bg=self.bg_panel)
        type_frame.pack(side=tk.TOP, fill=tk.X, pady=(15, 5), padx=10)
        tk.Label(type_frame, text="交互类型:", bg=self.bg_panel, fg=self.color_text, font=self.font_normal).pack(side=tk.LEFT)
        self._create_radio_button(type_frame, "点击", self.state['interact_type'], 'click').pack(side=tk.LEFT, padx=5)
        self._create_radio_button(type_frame, "涂鸦", self.state['interact_type'], 'scribble').pack(side=tk.LEFT, padx=5)

        # 2. 涂鸦区域选择 (新增)
        area_frame = tk.Frame(self.interact_frame, bg=self.bg_panel)
        area_frame.pack(side=tk.TOP, fill=tk.X, pady=5, padx=10)
        tk.Label(area_frame, text="涂鸦区域:", bg=self.bg_panel, fg=self.color_text, font=self.font_normal).pack(side=tk.LEFT)
        self._create_radio_button(area_frame, "前景", self.state['scribble_area'], 'foreground').pack(side=tk.LEFT, padx=2)
        self._create_radio_button(area_frame, "背景", self.state['scribble_area'], 'background').pack(side=tk.LEFT, padx=2)
        self._create_radio_button(area_frame, "未知", self.state['scribble_area'], 'unknown').pack(side=tk.LEFT, padx=2)

        # 3. 笔刷大小
        brush_frame = tk.Frame(self.interact_frame, bg=self.bg_panel)
        brush_frame.pack(side=tk.TOP, fill=tk.X, pady=5, padx=10)
        tk.Label(brush_frame, text="笔刷大小:", bg=self.bg_panel, fg=self.color_text, font=self.font_normal).pack(side=tk.TOP, anchor='w')
        s1 = FocusHorizontalScale(brush_frame, from_=1, to=25, resolution=1, variable=self.state['click_radius'], command=self._update_click_radius)
        s1.pack(fill=tk.X, pady=2)
        s1.configure(bg=self.bg_panel, highlightthickness=0, troughcolor='#E8E8E8', activebackground=self.bg_panel)

        # 4. 透明度
        alpha_frame = tk.Frame(self.interact_frame, bg=self.bg_panel)
        alpha_frame.pack(side=tk.TOP, fill=tk.X, pady=5, padx=10)
        tk.Label(alpha_frame, text="透明度:", bg=self.bg_panel, fg=self.color_text, font=self.font_normal).pack(side=tk.TOP, anchor='w')
        s2 = FocusHorizontalScale(alpha_frame, from_=0.0, to=1.0, variable=self.state['alpha_blend'], command=self._update_blend_alpha)
        s2.pack(fill=tk.X, pady=2)
        s2.configure(bg=self.bg_panel, highlightthickness=0, troughcolor='#E8E8E8', activebackground=self.bg_panel)

        # 分割线
        tk.Frame(self.interact_frame, bg='#F0F2F5', height=2).pack(fill=tk.X, padx=10, pady=5)

        # 5. 点击交互控制
        click_ctrl_frame = tk.Frame(self.interact_frame, bg=self.bg_panel)
        click_ctrl_frame.pack(side=tk.TOP, fill=tk.X, pady=5, padx=10)
        tk.Label(click_ctrl_frame, text="点击操作:", bg=self.bg_panel, fg=self.color_text, font=self.font_normal).pack(side=tk.LEFT)
        
        self.undo_click_btn = FocusButton(click_ctrl_frame, text="撤销点击", command=self.controller.undo_click, state=tk.DISABLED)
        self.undo_click_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(10, 5), ipady=2)
        self._style_button(self.undo_click_btn, 'warning') # 橙色突出

        self.reset_click_btn = FocusButton(click_ctrl_frame, text="重置点击", command=self._reset_last_object, state=tk.DISABLED)
        self.reset_click_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, ipady=2)
        self._style_button(self.reset_click_btn, 'danger') # 红色突出

        # 6. 涂鸦交互控制
        scribble_ctrl_frame = tk.Frame(self.interact_frame, bg=self.bg_panel)
        scribble_ctrl_frame.pack(side=tk.TOP, fill=tk.X, pady=10, padx=10)
        tk.Label(scribble_ctrl_frame, text="涂鸦操作:", bg=self.bg_panel, fg=self.color_text, font=self.font_normal).pack(side=tk.LEFT)
        
        self.undo_scribble_btn = FocusButton(scribble_ctrl_frame, text="撤销涂鸦", command=self.controller.undo_scribble, state=tk.DISABLED)
        self.undo_scribble_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(10, 5), ipady=2)
        self._style_button(self.undo_scribble_btn, 'warning') # 橙色突出

        self.reset_scribble_btn = FocusButton(scribble_ctrl_frame, text="重置涂鸦", command=self._reset_last_object, state=tk.DISABLED)
        self.reset_scribble_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, ipady=2)
        self._style_button(self.reset_scribble_btn, 'danger') # 红色突出

    def _add_canvas_video(self, r, c):
        self.canvas_frame_video, self.canvas_video = self._create_styled_canvas_frame(r, c, " 视频预览区域 ", "等待加载视频...")

    def _add_canvas_result(self, r, c):
        self.canvas_frame_result, self.canvas_result = self._create_styled_canvas_frame(r, c, " 结果预览区域 ", "等待生成结果...")

    def _add_video_controls(self, r, c):
        self.video_ctrl_frame = FocusLabelFrame(self.main_frame, text=" 视频控制栏 ", bg=self.bg_panel, font=self.font_title, fg=self.color_text)
        self.video_ctrl_frame.grid(row=r, column=c, sticky='nswe', padx=8, pady=8)

        tk.Frame(self.video_ctrl_frame, bg=self.bg_panel).pack(expand=True)

        v_play_frame = tk.Frame(self.video_ctrl_frame, bg=self.bg_panel)
        v_play_frame.pack(side=tk.TOP, fill=tk.X, pady=10, padx=15)
        
        btn_v_play = FocusButton(v_play_frame, text="▶ 视频播放", command=self._play_video)
        btn_v_play.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, ipady=4)
        self._style_button(btn_v_play, 'primary') # 蓝色突出

        btn_v_pause = FocusButton(v_play_frame, text="⏸ 视频暂停", command=self._pause_video)
        btn_v_pause.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, ipady=4)
        self._style_button(btn_v_pause, 'warning') # 橙色突出

        r_play_frame = tk.Frame(self.video_ctrl_frame, bg=self.bg_panel)
        r_play_frame.pack(side=tk.TOP, fill=tk.X, pady=10, padx=15)
        
        btn_r_play = FocusButton(r_play_frame, text="▶ 结果播放")
        btn_r_play.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, ipady=4)
        self._style_button(btn_r_play, 'primary') # 蓝色突出

        btn_r_pause = FocusButton(r_play_frame, text="⏸ 结果暂停")
        btn_r_pause.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, ipady=4)
        self._style_button(btn_r_pause, 'warning') # 橙色突出

        tk.Frame(self.video_ctrl_frame, bg=self.bg_panel).pack(expand=True)

        matting_frame = tk.Frame(self.video_ctrl_frame, bg=self.bg_panel)
        matting_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20, padx=20)
        btn_start_matting = FocusButton(matting_frame, text="开 始 抠 图")
        btn_start_matting.pack(fill=tk.X, ipady=8)
        self._style_button(btn_start_matting, 'success') # 绿色核心突出

    def _load_file_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("支持的文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.mp4 *.avi *.mov"),
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("视频文件", "*.mp4 *.avi *.mov"),
                ("所有文件", "*.*"),
            ], title="选择文件")

            if len(filename) > 0:
                # 判断是视频文件还是图像文件
                file_ext = os.path.splitext(filename)[1].lower()
                video_exts = ['.mp4', '.avi', '.mov']
                
                if file_ext in video_exts:
                    # 处理视频文件
                    self._load_video_file(filename)
                else:
                    # 处理图像文件
                    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
                    if image is not None:
                        self.canvas.delete("placeholder")
                        self.canvas_trimap.delete("placeholder")
                        
                        # 检查模型是否加载
                        if self.c2t_model is None:
                            messagebox.showwarning("警告", "请先加载 Click2Trimap 模型！")
                            return
                        
                        self.controller.set_image(image)
                        self.save_mask_btn.configure(state=tk.NORMAL)
                        self.save_video_btn.configure(state=tk.NORMAL)

    def _save_mask_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            mask = self.controller.result_mask
            trimap = self.controller.trimap_save

            if mask is None:
                return

            trimap_name = filedialog.asksaveasfilename(parent=self.master, initialfile='trimap.png', filetypes=[
                ("PNG 图像", "*.png"),
                ("BMP 图像", "*.bmp"),
                ("所有文件", "*.*"),
            ], title="保存三分图为...")

            if len(trimap_name) > 0:
                if mask.max() < 256:
                    mask = mask.astype(np.uint8)
                    mask *= 255 // mask.max()
                cv2.imwrite(trimap_name, trimap)

    def _save_video_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            video_name = filedialog.asksaveasfilename(parent=self.master, initialfile='result.mp4', filetypes=[
                ("MP4 视频", "*.mp4"),
                ("AVI 视频", "*.avi"),
                ("所有文件", "*.*"),
            ], title="保存结果视频为...")

    def _reset_last_object(self):
        self.state['alpha_blend'].set(0.5)
        self.state['prob_thresh'].set(0.5)
        self.controller.reset_last_object()

    def _update_prob_thresh(self, value):
        if self.controller.is_incomplete_mask:
            self.controller.prob_thresh = self.state['prob_thresh'].get()
            self._update_image()

    def _update_blend_alpha(self, value):
        self._update_image()

    def _update_click_radius(self, *args):
        if self.image_on_canvas is None:
            return
        self._update_image()

    def _reset_predictor(self, *args, **kwargs):
        brs_mode = 'NoBRS'
        prob_thresh = self.state['prob_thresh'].get()
        net_clicks_limit = None 

        if self.state['zoomin_params']['use_zoom_in'].get():
            zoomin_params = {
                'skip_clicks': self.state['zoomin_params']['skip_clicks'].get(),
                'target_size': self.state['zoomin_params']['target_size'].get(),
                'expansion_ratio': self.state['zoomin_params']['expansion_ratio'].get()
            }
            if self.state['zoomin_params']['fixed_crop'].get():
                zoomin_params['target_size'] = (zoomin_params['target_size'], zoomin_params['target_size'])
        else:
            zoomin_params = None

        predictor_params = {
            'brs_mode': brs_mode,
            'prob_thresh': prob_thresh,
            'zoom_in_params': zoomin_params,
            'predictor_params': {
                'net_clicks_limit': net_clicks_limit,
                'max_size': self.limit_longest_size
            },
            'brs_opt_func_params': {'min_iou_diff': 1e-3},
            'lbfgs_params': {'maxfun': self.state['lbfgs_max_iters'].get()}
        }
        self.controller.reset_predictor(predictor_params)

    def _click_callback(self, flag, x, y):
        self.canvas.focus_set()

        if self.image_on_canvas is None:
            messagebox.showwarning("提示", "请先加载图像或视频文件！")
            return

        if self._check_entry(self):
            if self.state['interact_type'].get() == 'click':
                self.controller.add_click(x, y, flag)

    def _scribble_callback(self, x, y):
        self.canvas.focus_set()

        if self.image_on_canvas is None:
            return

        if self._check_entry(self):
            if self.state['interact_type'].get() == 'scribble':
                if not self._is_scribbling:
                    self._is_scribbling = True
                    self._prev_scribble_x = None
                    self._prev_scribble_y = None
                    self.controller.start_scribble()
                
                area_type = self.state['scribble_area'].get()
                radius = self.state['click_radius'].get()
                self.controller.add_scribble(x, y, area_type, radius, self._prev_scribble_x, self._prev_scribble_y)
                
                self._prev_scribble_x = x
                self._prev_scribble_y = y

    def _scribble_release_callback(self):
        if self._is_scribbling:
            self._is_scribbling = False
            self._prev_scribble_x = None
            self._prev_scribble_y = None

    def _update_image(self, reset_canvas=False, backgroung_flag=None):
        image, trimap, image_source = self.controller.get_visualization(alpha_blend=self.state['alpha_blend'].get(),
                                                  click_radius=self.state['click_radius'].get())

        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas)
            self.image_on_canvas.register_click_callback(self._click_callback)
            self.image_on_canvas.register_scribble_callbacks(self._scribble_callback, self._scribble_release_callback)

        if self.trimap_on_canvas is None:
            self.trimap_on_canvas = CanvasImage(self.canvas_frame_trimap, self.canvas_trimap)
            self.trimap_on_canvas.register_click_callback(self._click_callback)
            self.trimap_on_canvas.register_scribble_callbacks(self._scribble_callback, self._scribble_release_callback)

        self._set_click_dependent_widgets_state()
        if image is not None:
            self.image_on_canvas.reload_image(Image.fromarray(image), reset_canvas)
            
        if trimap is not None:
            self.trimap_on_canvas.reload_image(Image.fromarray(trimap), reset_canvas)
        else:
            black = np.zeros((400, 800), dtype=np.uint8) if image is None else np.zeros(image.shape[:2], dtype=np.uint8)
            self.trimap_on_canvas.reload_image(Image.fromarray(black), reset_canvas)
    
    def _set_click_dependent_widgets_state(self):
        # 点击相关按钮：基于点击历史
        has_click_history = len(self.controller.click_probs_history) > 0
        click_state = tk.NORMAL if has_click_history else tk.DISABLED
        
        # 涂鸦相关按钮：基于涂鸦历史
        has_scribble_history = len(self.controller.scribble_probs_history) > 0
        scribble_state = tk.NORMAL if has_scribble_history else tk.DISABLED
        
        # 任何历史存在时启用重置按钮
        has_any_history = has_click_history or has_scribble_history
        reset_state = tk.NORMAL if has_any_history else tk.DISABLED

        self.undo_click_btn.configure(state=click_state)
        self.reset_click_btn.configure(state=reset_state)
        self.undo_scribble_btn.configure(state=scribble_state)
        self.reset_scribble_btn.configure(state=reset_state)

    def _check_entry(self, widget):
        all_checked = True
        if widget.winfo_children is not None:
            for w in widget.winfo_children():
                all_checked = all_checked and self._check_entry(w)

        if getattr(widget, "_check_bounds", None) is not None:
            all_checked = all_checked and widget._check_bounds(widget.get(), '-1')

        return all_checked

    # ==================== 视频处理相关方法 ====================
    
    def _load_video_file(self, video_path):
        """加载视频文件并启动拆帧处理"""
        # 检查模型是否加载
        if self.c2t_model is None:
            messagebox.showwarning("警告", "请先加载 Click2Trimap 模型！")
            return
        
        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(video_path)
        
        if not self.video_capture.isOpened():
            messagebox.showerror("错误", "无法打开视频文件！")
            return
        
        # 获取视频总帧数
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            messagebox.showerror("错误", "无法获取视频帧数！")
            return
        
        # 获取视频FPS
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.video_fps = int(fps) if fps > 0 else 30
        
        # 创建拆帧保存目录
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        unique_id = str(uuid.uuid4())[:8]
        self.frames_folder = os.path.join('trimap_video_frames', f'{video_name}_{unique_id}')
        os.makedirs(self.frames_folder, exist_ok=True)
        
        # 创建进度条窗口
        self._create_progress_window(total_frames)
        
        # 释放当前捕获对象，在后台线程中重新打开
        self.video_capture.release()
        
        # 在后台线程中进行拆帧处理
        thread = threading.Thread(target=self._extract_video_frames, 
                                   args=(video_path, total_frames, unique_id))
        thread.daemon = True
        thread.start()

    def _create_progress_window(self, total_frames):
        """创建进度条窗口"""
        # 先隐藏进度条窗口如果已存在
        if hasattr(self, 'progress_window') and self.progress_window is not None:
            try:
                self.progress_window.destroy()
            except:
                pass
        
        # 创建进度条窗口
        self.progress_window = tk.Toplevel(self.master)
        self.progress_window.title("视频拆帧处理")
        self.progress_window.geometry("400x120")
        self.progress_window.resizable(False, False)
        
        # 居中显示
        self.progress_window.transient(self.master)
        self.progress_window.grab_set()
        
        # 进度条框架
        progress_frame = tk.Frame(self.progress_window, bg=self.bg_panel)
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 提示标签
        self.progress_label = tk.Label(progress_frame, text="正在拆帧: 0/{0}".format(total_frames),
                                       bg=self.bg_panel, fg=self.color_text, font=self.font_normal)
        self.progress_label.pack(pady=(0, 10))
        
        # 进度条
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=350)
        self.progress_bar.pack()
        self.progress_bar['maximum'] = total_frames
        self.progress_bar['value'] = 0
        
        # 百分比标签
        self.progress_percent = tk.Label(progress_frame, text="0%",
                                         bg=self.bg_panel, fg=self.color_text, font=self.font_normal)
        self.progress_percent.pack(pady=(10, 0))

    def _extract_video_frames(self, video_path, total_frames, unique_id):
        """在后台线程中提取视频帧并保存"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.master.after(0, lambda: messagebox.showerror("错误", "无法打开视频文件！"))
            return
        
        self.video_frames = []
        frames_saved = []
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR 转 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.video_frames.append(frame_rgb)
            
            # 保存帧到文件
            frame_filename = os.path.join(self.frames_folder, f'frame_{i:06d}.png')
            cv2.imwrite(frame_filename, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            frames_saved.append(frame_filename)
            
            # 更新进度条（每10帧更新一次以减少UI负担）
            if i % 10 == 0 or i == total_frames - 1:
                self.master.after(0, lambda idx=i: self._update_progress(idx + 1, total_frames))
        
        cap.release()
        
        # 拆帧完成后在主线程中更新UI
        self.master.after(0, lambda: self._on_frames_extracted(frames_saved))

    def _update_progress(self, current, total):
        """更新进度条显示"""
        if hasattr(self, 'progress_bar') and self.progress_bar is not None:
            self.progress_bar['value'] = current
            percent = int(current / total * 100)
            if hasattr(self, 'progress_label'):
                self.progress_label.config(text="正在拆帧: {0}/{1}".format(current, total))
            if hasattr(self, 'progress_percent'):
                self.progress_percent.config(text="{0}%".format(percent))
            self.progress_window.update()

    def _on_frames_extracted(self, frames_saved):
        """帧提取完成后的回调"""
        # 关闭进度条窗口
        if hasattr(self, 'progress_window') and self.progress_window is not None:
            try:
                self.progress_window.destroy()
            except:
                pass
            self.progress_window = None
        
        if len(self.video_frames) == 0:
            messagebox.showerror("错误", "视频帧提取失败！")
            return
        
        # 在图像画布显示第一帧
        first_frame = self.video_frames[0]
        self.canvas.delete("placeholder")
        self.canvas_trimap.delete("placeholder")
        self.controller.set_image(first_frame)
        
        # 在视频预览区域显示第一帧
        self._display_video_frame(0)
        
        # 启用保存按钮
        self.save_mask_btn.configure(state=tk.NORMAL)
        self.save_video_btn.configure(state=tk.NORMAL)
        
        # 显示完成消息
        messagebox.showinfo("完成", "视频拆帧完成！共提取 {0} 帧，保存至: {1}".format(
            len(self.video_frames), self.frames_folder))

    def _display_video_frame(self, frame_index):
        """在视频预览区域显示指定帧"""
        if frame_index >= len(self.video_frames):
            frame_index = 0
        
        self.current_frame_index = frame_index
        frame = self.video_frames[frame_index]
        
        # 调整图像大小以适应画布
        canvas_width = self.canvas_video.winfo_width()
        canvas_height = self.canvas_video.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # 画布还未完全渲染，使用默认大小
            canvas_width = 480
            canvas_height = 270
        
        h, w = frame.shape[:2]
        scale = min(canvas_width / w, canvas_height / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 调整图像大小
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 转换为 PhotoImage
        frame_pil = Image.fromarray(frame_resized)
        self.video_photo = ImageTk.PhotoImage(frame_pil)
        
        # 显示图像
        self.canvas_video.delete("all")
        x = (canvas_width - new_w) // 2
        y = (canvas_height - new_h) // 2
        self.canvas_video.create_image(x, y, anchor=tk.NW, image=self.video_photo)
        
        # 清除占位符
        self.canvas_video.delete("placeholder")

    def _play_video(self):
        """播放视频"""
        if self.video_frames is None or len(self.video_frames) == 0:
            messagebox.showwarning("提示", "请先加载视频文件！")
            return
        
        if self.is_playing:
            return
        
        self.is_playing = True
        
        # 使用之前保存的FPS或默认值
        fps = self.video_fps if hasattr(self, 'video_fps') and self.video_fps > 0 else 30
        delay = int(1000 / fps)  # 每帧延迟毫秒数
        
        def play_loop():
            while self.is_playing and self.current_frame_index < len(self.video_frames) - 1:
                self.current_frame_index += 1
                self.master.after(0, lambda idx=self.current_frame_index: self._display_video_frame(idx))
                self.master.after(delay)
            
            # 播放完成
            if self.current_frame_index >= len(self.video_frames) - 1:
                self.is_playing = False
        
        self.video_play_thread = threading.Thread(target=play_loop)
        self.video_play_thread.daemon = True
        self.video_play_thread.start()

    def _pause_video(self):
        """暂停视频播放"""
        self.is_playing = False

# --- END OF FILE app6.py ---
