import torch
import numpy as np
from tkinter import messagebox
import cv2
from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks
# from inference import single_inference, generator_tensor_dict
import time

def single_inference(image,trimap,model):

    trimap_nonp=trimap.copy()
    h,w,c=image.shape
    nonph,nonpw,_=image.shape
    newh= (((h-1)//32)+1)*32
    neww= (((w-1)//32)+1)*32
    padh=newh-h
    padh1=int(padh/2)
    padh2=padh-padh1
    padw=neww-w
    padw1=int(padw/2)
    padw2=padw-padw1
    image_pad=cv2.copyMakeBorder(image,padh1,padh2,padw1,padw2,cv2.BORDER_REFLECT)
    trimap_pad=cv2.copyMakeBorder(trimap,padh1,padh2,padw1,padw2,cv2.BORDER_REFLECT)
    h_pad,w_pad,_=image_pad.shape
    tritemp = np.zeros([*trimap_pad.shape, 3], np.float32)
    tritemp[:, :, 0] = (trimap_pad == 0)
    tritemp[:, :, 1] = (trimap_pad == 128)
    tritemp[:, :, 2] = (trimap_pad == 255)
    tritempimgs=np.transpose(tritemp,(2,0,1))
    tritempimgs=tritempimgs[np.newaxis,:,:,:]
    img=np.transpose(image_pad,(2,0,1))[np.newaxis,::-1,:,:]
    img=np.array(img,np.float32)
    img=img/255.
    img=torch.from_numpy(img).cuda()
    tritempimgs=torch.from_numpy(tritempimgs).cuda()

    with torch.no_grad():
        # time1 = time.time()
        pred=model(img,tritempimgs)
        # print('time_Matting:',time.time()-time1)
        pred=pred.detach().cpu().numpy()[0]
        pred=pred[:,padh1:padh1+h,padw1:padw1+w]
        preda=pred[0:1,]*255
        preda=np.transpose(preda,(1,2,0))
        preda=preda*(trimap_nonp[:,:,None]==128)+(trimap_nonp[:,:,None]==255)*255
    preda=np.array(preda,np.uint8)

    return preda

class InteractiveController:
    def __init__(self, net, device, predictor_params, update_image_callback, prob_thresh=0.5):
        self.net = net
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.click_states = []
        self.scribble_states = []
        self.click_probs_history = []
        self.scribble_probs_history = []
        self.object_count = 0
        self._result_mask = None
        self._init_mask = None

        self.image = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.reset_predictor()

    def set_image(self, image):
        self.image = image
        self._result_mask = np.zeros_like(image, dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        self.update_image_callback(reset_canvas=True)

    def set_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
        self.clicker.click_indx_offset = 1

    def add_click(self, x, y, flag):
        self.click_states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
        })

        click = clicker.Click(flag=flag, coords=(y, x))
        self.clicker.add_click(click)
        pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)

        if self._init_mask is not None and len(self.clicker) == 1:
            pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)

        torch.cuda.empty_cache()

        if self.click_probs_history:
            self.click_probs_history.append((self.click_probs_history[-1][0], pred))
        else:
            self.click_probs_history.append((np.zeros_like(pred), pred))

        self.update_image_callback()

    def start_scribble(self):
        # 使用涂鸦专用历史记录
        if not self.scribble_probs_history:
            # 检查是否有点击历史，如果有则基于点击结果
            if self.click_probs_history:
                # 拷贝点击的完整状态：累积概率(total)和增量概率(additive)
                click_total = self.click_probs_history[-1][0].copy()
                click_additive = self.click_probs_history[-1][1].copy()
                self.scribble_probs_history.append((click_total, click_additive))
            else:
                pred = np.zeros((3, self.image.shape[0], self.image.shape[1]), dtype=np.float32)
                pred[0, :, :] = 1000 # default background
                self.scribble_probs_history.append((np.zeros_like(pred), pred))
        else:
            current_prob_total, current_prob_additive = self.scribble_probs_history[-1]
            self.scribble_probs_history.append((current_prob_total.copy(), current_prob_additive.copy()))
        
        self.scribble_states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
        })

    def add_scribble(self, x, y, area_type, radius, prev_x=None, prev_y=None):
        if not self.scribble_probs_history:
            return
        
        # 使用涂鸦专用历史记录
        current_prob_total, current_prob_additive = self.scribble_probs_history[-1]
        
        H, W = current_prob_additive.shape[1:]
        
        # Create a mask for the scribble
        mask = np.zeros((H, W), dtype=bool)
        
        if prev_x is not None and prev_y is not None:
            # Draw a line from previous point to current point
            # We use cv2.line to draw a thick line on a temporary mask
            temp_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.line(temp_mask, (int(prev_x), int(prev_y)), (int(x), int(y)), 1, thickness=int(radius*2))
            mask = temp_mask > 0
        else:
            # Just draw a circle at the current point
            Y, X = np.ogrid[:H, :W]
            dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
            mask = dist_from_center <= radius
        
        if area_type == 'foreground':
            idx = 2
        elif area_type == 'unknown':
            idx = 1
        else: # background
            idx = 0
            
        for i in range(3):
            if i == idx:
                current_prob_additive[i, mask] = 1000
            else:
                current_prob_additive[i, mask] = -1000
                
        self.update_image_callback()

    def undo_click(self):
        """仅撤销点击，不撤销涂鸦"""
        if not self.click_states:
            return

        prev_state = self.click_states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
        self.click_probs_history.pop()
        if not self.click_probs_history:
            self.reset_init_mask()
        self.update_image_callback()

    def undo_scribble(self):
        """仅撤销涂鸦，不撤销点击"""
        if not self.scribble_states:
            return

        prev_state = self.scribble_states.pop()
        self.scribble_probs_history.pop()
        self.update_image_callback()

    def change_background_1(self):
        self.update_image_callback(backgroung_flag=1)

    def change_background_2(self):
        self.update_image_callback(backgroung_flag=2)

    def change_background_3(self):
        self.update_image_callback(backgroung_flag=3)

    def partially_finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        # 使用当前的历史记录类型
        if self.scribble_probs_history:
            self.scribble_probs_history.append((object_prob, np.zeros_like(object_prob)))
            self.scribble_states.append(self.scribble_states[-1])
        else:
            self.click_probs_history.append((object_prob, np.zeros_like(object_prob)))
            self.click_states.append(self.click_states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        self.update_image_callback()

    def finish_object(self):
        if self.current_object_prob is None:
            return

        self._result_mask = self.result_mask
        self.object_count += 1
        self.reset_last_object()

    def reset_last_object(self, update_image=True):
        self.click_states = []
        self.scribble_states = []
        self.click_probs_history = []
        self.scribble_probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        if update_image:
            self.update_image_callback()

    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self._init_mask = None
        self.clicker.click_indx_offset = 0

    @property
    def probs_history(self):
        """合并点击和涂鸦的历史记录用于显示"""
        if self.scribble_probs_history:
            return self.scribble_probs_history
        return self.click_probs_history

    @property
    def states(self):
        """合并点击和涂鸦的状态用于兼容"""
        if self.scribble_states:
            return self.scribble_states
        return self.click_states

    @property
    def current_object_prob(self):
        # 使用涂鸦历史或点击历史
        if self.scribble_probs_history:
            current_prob_total, current_prob_additive = self.scribble_probs_history[-1]
        elif self.click_probs_history:
            current_prob_total, current_prob_additive = self.click_probs_history[-1]
        else:
            return None

        fore_mask = current_prob_additive.argmax(axis=0) == 2
        uk_mask = current_prob_additive.argmax(axis=0) == 1
        back_mask = current_prob_additive.argmax(axis=0) == 0

        scribbled_back_mask = (current_prob_additive[0] == 1000) & (current_prob_additive[1] == -1000)
        back_mask = back_mask & (~scribbled_back_mask)

        pred_mask = np.stack([back_mask, uk_mask, fore_mask, scribbled_back_mask]).astype(np.uint8)
        return pred_mask

    @property
    def is_incomplete_mask(self):
        return len(self.click_probs_history) > 0 or len(self.scribble_probs_history) > 0

    @property
    def result_mask(self):
        if self.probs_history:
            return self.current_object_prob
        return None

    def get_visualization(self, alpha_blend, click_radius):
        if self.image is None:
            return None

        trimap = None

        results_mask_for_vis = self.result_mask
        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)

        if results_mask_for_vis is not None:
            trimap = results_mask_for_vis[0] * 0 + results_mask_for_vis[1] * 128 + results_mask_for_vis[2] * 255

        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            if total_mask.shape[0] == 3 and results_mask_for_vis.shape[0] == 4:
                total_mask = np.concatenate([total_mask, np.zeros_like(total_mask[:1])], axis=0)
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend)

            if hasattr(self, 'matting_model') and self.matting_model is not None:
                alpha = single_inference(self.image, trimap, self.matting_model)
            else:
                alpha = None
            
            self.alpha_save = alpha
        
        self.trimap_save = trimap
        return vis, trimap, self.image
