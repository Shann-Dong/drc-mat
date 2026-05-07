import os
import argparse
import time
import timeit
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image
import tqdm

from config import get_cfg_defaults
from dataset import EvalDataset, VideoMatting108_Test, Demo_Test
from helpers import *

torch.set_grad_enabled(False)

EPS = 0

'''增加交互式调用接口的推理代码'''

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument('--trimap', default='medium', choices=['narrow', 'medium', 'wide'])
    parser.add_argument("--viz", action='store_true')
    parser.add_argument("--demo", action='store_true')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.TRAIN.STAGE = 4

    if args.demo:
        cfg.SYSTEM.OUTDIR = './demo_results'
        cfg.DATASET.PATH = './demo'

    return args, cfg


def main(cfg, args, GPU):
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    MODEL = get_model_name(cfg)
    random_seed = cfg.SYSTEM.RANDOM_SEED
    output_dir = os.path.join(cfg.SYSTEM.OUTDIR, 'alpha')
    start = timeit.default_timer()
    cudnn.benchmark = False
    cudnn.deterministic = cfg.SYSTEM.CUDNN_DETERMINISTIC
    cudnn.enabled = cfg.SYSTEM.CUDNN_ENABLED
    if random_seed > 0:
        import random
        print('Seeding with', random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    if args.demo:
        outdir_tail = MODEL
    else:
        outdir_tail = os.path.join(args.trimap, MODEL)
    alpha_outdir = os.path.join(output_dir, 'test', outdir_tail)
    viz_outdir_img = os.path.join(output_dir, 'viz', 'img', outdir_tail)
    viz_outdir_vid = os.path.join(output_dir, 'viz', 'vid', outdir_tail)

    if args.trimap == 'narrow':
        dilate_kernel = 5   # width: 11
    elif args.trimap == 'medium':
        dilate_kernel = 12  # width: 25
    elif args.trimap == 'wide':
        dilate_kernel = 20  # width: 41

    model_trimap = get_model_trimap(cfg, mode='Test', dilate_kernel=dilate_kernel)
    model = get_model_alpha(cfg, model_trimap, mode='Test', dilate_kernel=dilate_kernel)
    
    load_ckpt = os.path.join('weights', '{:s}.pth'.format(MODEL))
    dct = torch.load(load_ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(dct)
    model = nn.DataParallel(model.cuda())


    if args.demo:
        valid_dataset = Demo_Test(data_root=cfg.DATASET.PATH)
    else:
        valid_dataset = VideoMatting108_Test(
            data_root=cfg.DATASET.PATH,
            mode='val',
        )
    with torch.no_grad():
        eval(args, cfg, valid_dataset, model, alpha_outdir, viz_outdir_img, viz_outdir_vid, args.viz)
    
    end = timeit.default_timer()
    print('done | Total time: {}'.format(format_time(end-start)))

def write_image(outdir, out, filename, max_batch=4):
    with torch.no_grad():
        scaled_imgs, tri_pred, tri_gt, alphas, scaled_gts, comps = out
        b, s, _, h, w = scaled_imgs.shape
        alphas = alphas.expand(-1,-1,3,-1,-1)
        scaled_gts = scaled_gts.expand(-1,-1,3,-1,-1)

        b = max_batch if b > max_batch else b
        img_list = list()
        img_list.append(scaled_imgs[:max_batch].reshape(b*s, 3, h, w))
        img_list.append(comps[:max_batch].reshape(b*s, 3, h, w))
        img_list.append(tri_gt[:max_batch].reshape(b*s, 3, h, w))
        img_list.append(scaled_gts[:max_batch].reshape(b*s, 3, h, w))
        img_list.append(tri_pred[:max_batch].reshape(b*s, 3, h, w))
        img_list.append(alphas[:max_batch].reshape(b*s, 3, h, w))
        imgs = torch.cat(img_list, dim=0).reshape(-1, 3, h, w)

        imgs = F.interpolate(imgs, size=(h//2, w//2), mode='bilinear', align_corners=False)
        
        save_image(imgs, outdir%(filename), nrow=int(s*b*2))

def eval(args, cfg, valid_dataset, model, alpha_outdir, viz_outdir_img, viz_outdir_vid, VIZ):
    model.eval()

    for i_iter, (data_name, data_root, FG, BG, a, tri, seq_name) in enumerate(valid_dataset):
        if cfg.SYSTEM.TESTMODE:
            if i_iter not in [0, len(valid_dataset)-1]:
                continue
        torch.cuda.empty_cache()
        num_frames = 1
        eval_sequence = EvalDataset(
            data_name=data_name,
            data_root=data_root,
            FG=FG,
            BG=BG,
            a=a,
            tri_gt=tri, # GT trimap
            trimap=None,
            num_frames=num_frames,
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_sequence,
            batch_size=1,
            # num_workers=cfg.SYSTEM.NUM_WORKERS,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
            sampler=None)

        print('[{}/{}] Set FIXED dilate of unknown region: [{}]'.format(i_iter, len(valid_dataset), args.trimap))

        save_path = os.path.join(alpha_outdir, 'pred', seq_name)
        os.makedirs(save_path, exist_ok=True)
        if VIZ:
            visualization_path_img = os.path.join(viz_outdir_img, 'viz', seq_name)
            visualization_path_vid = os.path.join(viz_outdir_vid, 'viz')
            os.makedirs(visualization_path_img, exist_ok=True)
            os.makedirs(visualization_path_vid, exist_ok=True)

        iterations = tqdm.tqdm(eval_loader)
        for i_seq, dp in enumerate(iterations):
            if cfg.SYSTEM.TESTMODE:
                if i_seq > 10:
                    break

            def handle_batch(dp, first_frame, last_frame, memorize, max_memory_num, large_input):
                fg, bg, a, eps, tri_gt, tri, _, filename = dp      # [B, 3, 3 or 1, H, W]

                if tri.dim() == 1:
                    tri = None
                if tri_gt.dim() == 1:
                    tri_gt = None
                
                out = model(a, fg, bg, tri=tri, tri_gt=tri_gt,
                            first_frame=first_frame,
                            last_frame=last_frame,
                            memorize=memorize,
                            max_memory_num=max_memory_num,
                            large_input=large_input,)
                return out, filename[0]

            first_frame = (i_seq==0)
            last_frame = (i_seq==(len(iterations)-1))
            memorize = False
            MEMORY_SKIP_FRAME = cfg.TEST.MEMORY_SKIP_FRAME
            MEMORY_MAX_NUM = cfg.TEST.MEMORY_MAX_NUM
            large_input = False
            if min(dp[0].shape[-2:]) > 1100:
                MEMORY_SKIP_FRAME = int(MEMORY_SKIP_FRAME * 2)
                MEMORY_MAX_NUM = int(MEMORY_MAX_NUM / 2)
                large_input = True
            if MEMORY_SKIP_FRAME > 2:
                memorize = (i_seq % MEMORY_SKIP_FRAME) == 0
            max_memory_num = MEMORY_MAX_NUM
            
            if first_frame:
                print('[{}/{}] {} | {} | Large input: {}'.format(i_iter, len(valid_dataset), seq_name, dp[0].shape[-2:], large_input))
                
            torch.cuda.synchronize()
            out, filename = handle_batch(dp, first_frame, last_frame, memorize, max_memory_num, large_input,)
            torch.cuda.synchronize()

            scaled_imgs, tri_pred, tri_gt, alphas, scaled_gts = out

            green_bg = torch.zeros_like(scaled_imgs)
            green_bg[:,:,1] = 1.
            comps = scaled_imgs * alphas + green_bg * (1. - alphas)
            
            if VIZ:
                frame_path = os.path.join(visualization_path_img, 'f%d.jpg')
            else:
                frame_path = None
            alpha_pred_img = (alphas*255).byte().cpu().squeeze(0).squeeze(0).squeeze(0).numpy()
            filename_for_save = os.path.splitext(filename)[0]+'.png'

            def write_result_images(alpha_pred_img, path, VIZ, frame_path, vis_out, i_seq):
                if VIZ:
                    write_image(frame_path,
                                vis_out,
                                i_seq)
                cv2.imwrite(path, alpha_pred_img)
            
            write_result_images(alpha_pred_img,
                                os.path.join(save_path, filename_for_save),
                                VIZ,
                                frame_path,
                                # [scaled_imgs, tri_pred, tri_gt, alphas, scaled_gts, comps],
                                [scaled_imgs.cpu(), tri_pred.cpu(), tri_gt.cpu(), alphas.cpu(), scaled_gts.cpu(), comps.cpu()],
                                i_seq)


            torch.cuda.synchronize()
        
        if VIZ:
            if '/' in seq_name:
                vid_name = seq_name.split('/')
                vid_name = '_'.join(vid_name)
            else:
                vid_name = seq_name
            vid_path = os.path.join(visualization_path_vid, '{}.mp4'.format(vid_name))

            def make_viz_video(frame_path, vid_path):
                os.system('ffmpeg -framerate 10 -i {} {}  -nostats -loglevel 0 -y'.format(frame_path, vid_path))
                time.sleep(10) # wait 10 seconds

            make_viz_video(frame_path, vid_path)

def run_inference(frames_folder, trimap_folder, output_dir, gpu='0', progress_callback=None):
    """
    外部调用接口：对指定视频帧和三分图目录执行 OTVM 推理，生成 alpha matte 帧序列并合成结果视频。

    参数
    ----
    frames_folder  : str  视频帧目录（PNG/JPG 图像，文件名按字典序排列即为帧顺序）
    trimap_folder  : str  三分图目录（与视频帧同名，扩展名为 .png）
    output_dir     : str  推理结果的根输出目录
    gpu            : str  GPU ID 字符串，默认 '0'；无 GPU 时传 '-1' 使用 CPU
    progress_callback : callable | None  每处理一帧时回调 (current_frame, total_frames)

    返回
    ----
    result_video_path : str  合成的结果 MP4 视频路径（绿幕合成）；失败时返回 None
    alpha_dir         : str  逐帧 alpha matte 保存目录
    """
    import sys
    import shutil

    # ---- 配置 ----
    cfg = get_cfg_defaults()
    cfg.TRAIN.STAGE = 4

    # 设置 CUDA 可见设备
    if gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # ---- 模型名称与权重路径 ----
    MODEL = get_model_name(cfg)
    # 权重路径相对于 OTVM 目录，需确保当前工作目录正确
    otvm_dir = os.path.dirname(os.path.abspath(__file__))
    load_ckpt = os.path.join(otvm_dir, 'weights', '{:s}.pth'.format(MODEL))
    if not os.path.isfile(load_ckpt):
        raise FileNotFoundError('OTVM 权重文件不存在: {}'.format(load_ckpt))

    dilate_kernel = 12   # medium trimap width

    model_trimap = get_model_trimap(cfg, mode='Test', dilate_kernel=dilate_kernel)
    model = get_model_alpha(cfg, model_trimap, mode='Test', dilate_kernel=dilate_kernel)

    dct = torch.load(load_ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(dct)

    if torch.cuda.is_available() and gpu != '-1':
        model = nn.DataParallel(model.cuda())
    else:
        model = nn.DataParallel(model)   # CPU 模式

    model.eval()

    # ---- 构建帧列表 ----
    supported_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    frame_files = sorted([
        f for f in os.listdir(frames_folder)
        if os.path.splitext(f)[1].lower() in supported_exts
    ])
    if len(frame_files) == 0:
        raise ValueError('frames_folder 中没有图像文件: {}'.format(frames_folder))

    total_frames = len(frame_files)

    # ---- 输出目录 ----
    alpha_dir = os.path.join(output_dir, 'alpha_frames')
    os.makedirs(alpha_dir, exist_ok=True)

    # ---- 逐帧推理 ----
    with torch.no_grad():
        for i_seq, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frames_folder, frame_file)
            stem = os.path.splitext(frame_file)[0]

            # 加载原始帧（BGR）
            fg_bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if fg_bgr is None:
                print('警告：无法读取帧 {}'.format(frame_path))
                continue
            fg = torch.from_numpy(
                np.float32(fg_bgr).transpose(2, 0, 1)
            ).unsqueeze(0)   # [1, 3, H, W]

            # 加载三分图
            tri_path = os.path.join(trimap_folder, stem + '.png')
            if not os.path.isfile(tri_path):
                # 尝试同名 jpg
                tri_path = os.path.join(trimap_folder, stem + '.jpg')
            if not os.path.isfile(tri_path):
                # 使用目录中第一张三分图代替（允许只有首帧三分图的场景）
                existing = sorted([
                    f for f in os.listdir(trimap_folder)
                    if os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg'}
                ])
                if existing:
                    tri_path = os.path.join(trimap_folder, existing[0])
                else:
                    raise FileNotFoundError('三分图目录中没有图像文件: {}'.format(trimap_folder))

            tri_bgr = cv2.imread(tri_path, cv2.IMREAD_COLOR)
            if tri_bgr is None:
                raise FileNotFoundError('无法读取三分图: {}'.format(tri_path))

            # 将三分图转换为 one-hot [1, 3, H, W]：0=背景, 128=未知, 255=前景
            tri_gray = cv2.cvtColor(tri_bgr, cv2.COLOR_BGR2GRAY)
            H, W = tri_gray.shape
            tri_onehot = np.zeros((3, H, W), dtype=np.float32)
            tri_onehot[0][tri_gray < 85]  = 1.0   # background
            tri_onehot[1][(tri_gray >= 85) & (tri_gray < 170)] = 1.0  # unknown
            tri_onehot[2][tri_gray >= 170] = 1.0   # foreground
            tri_tensor = torch.from_numpy(tri_onehot).unsqueeze(0)  # [1, 3, H, W]

            # bg 设为与 fg 相同（无背景信息场景）
            bg = fg.clone()

            # alpha GT 占位
            a = torch.zeros(1, 1, H, W)
            tri_gt = torch.zeros(1, 3, H, W)
            eps = torch.tensor([0.])

            first_frame = (i_seq == 0)
            last_frame  = (i_seq == total_frames - 1)
            memorize    = False
            MEMORY_SKIP_FRAME = cfg.TEST.MEMORY_SKIP_FRAME
            MEMORY_MAX_NUM    = cfg.TEST.MEMORY_MAX_NUM
            large_input = False
            if min(H, W) > 1100:
                MEMORY_SKIP_FRAME = int(MEMORY_SKIP_FRAME * 2)
                MEMORY_MAX_NUM    = int(MEMORY_MAX_NUM / 2)
                large_input = True
            if MEMORY_SKIP_FRAME > 2:
                memorize = (i_seq % MEMORY_SKIP_FRAME) == 0

            if torch.cuda.is_available() and gpu != '-1':
                fg      = fg.cuda()
                bg      = bg.cuda()
                a       = a.cuda()
                tri_gt  = tri_gt.cuda()
                tri_tensor = tri_tensor.cuda()

            torch.cuda.synchronize() if (torch.cuda.is_available() and gpu != '-1') else None

            out = model(a, fg, bg,
                        tri=tri_tensor,
                        tri_gt=None,
                        first_frame=first_frame,
                        last_frame=last_frame,
                        memorize=memorize,
                        max_memory_num=MEMORY_MAX_NUM,
                        large_input=large_input)

            torch.cuda.synchronize() if (torch.cuda.is_available() and gpu != '-1') else None

            scaled_imgs, tri_pred, tri_gt_out, alphas, scaled_gts = out

            # 保存 alpha
            alpha_np = (alphas * 255).byte().cpu().squeeze().numpy()
            alpha_save_path = os.path.join(alpha_dir, stem + '.png')
            cv2.imwrite(alpha_save_path, alpha_np)

            if progress_callback is not None:
                try:
                    progress_callback(i_seq + 1, total_frames)
                except Exception:
                    pass

    # ---- 合成绿幕结果视频 ----
    result_video_path = os.path.join(output_dir, 'result_greenscreen.mp4')
    _compose_result_video(frames_folder, alpha_dir, frame_files, result_video_path)

    return result_video_path, alpha_dir


def _compose_result_video(frames_folder, alpha_dir, frame_files, output_video_path, fps=25):
    """将原始帧与 alpha matte 合成绿幕视频并写出 MP4。"""
    if len(frame_files) == 0:
        return

    # 读取第一帧确定尺寸
    first_frame_bgr = cv2.imread(os.path.join(frames_folder, frame_files[0]), cv2.IMREAD_COLOR)
    if first_frame_bgr is None:
        return
    H, W = first_frame_bgr.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    for frame_file in frame_files:
        stem = os.path.splitext(frame_file)[0]
        frame_bgr = cv2.imread(os.path.join(frames_folder, frame_file), cv2.IMREAD_COLOR)
        alpha_path = os.path.join(alpha_dir, stem + '.png')

        if frame_bgr is None:
            continue

        if os.path.isfile(alpha_path):
            alpha_gray = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
            if alpha_gray is not None:
                # 合成绿幕背景
                alpha_f = alpha_gray.astype(np.float32) / 255.0
                green_bg = np.zeros_like(frame_bgr, dtype=np.float32)
                green_bg[:, :, 1] = 255.0   # 绿色 (BGR: G=1)
                frame_f  = frame_bgr.astype(np.float32)
                alpha_3  = alpha_f[:, :, np.newaxis]
                comp     = (frame_f * alpha_3 + green_bg * (1.0 - alpha_3)).astype(np.uint8)
                writer.write(comp)
                continue

        # 无 alpha 则直接写原帧
        writer.write(frame_bgr)

    writer.release()


if __name__ == "__main__":
    args, cfg = parse_args()
    main(cfg, args, args.gpu)
