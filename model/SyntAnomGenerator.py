import imgaug.augmenters as iaa
import numpy as np
import torch
from einops import rearrange

from dataset.BGMaskDataset import BGMaskDataset
from dataset.SyntAnomDataset import SyntAnomDataset
from utils import MathUtils
from utils.geo_utils import *


class SyntAnomalyGenerator:
    def __init__(self, config):

        # General mask generation parameters
        self.image_width = config["img_width"]
        self.image_height = config["img_height"]
        self.backgrounds = SyntAnomDataset(config["path"], self.image_width, self.image_height)  # DTD
        self.use_pretrained_masks = config["use_pretrained_masks"]
        self.is_texture = config["is_texture"]

        # Extra params - some of them could be removed
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.structure_grid_size = [8, 8]
        self.resize = [self.image_width, self.image_height]

        path = config["fg_mask_path"]
        dataset = config["dataset"] + "_masks"
        path = path + "/" + dataset + "/" + config["category"] + "/train/good/"
        self.maskDataset = BGMaskDataset(path, self.image_width, self.image_height)

        self.rgb = "rgb" in config["mode"]
        self.d = "d" in config["mode"]

        self.lmax = 0.1

        # Add downsampling parameters
        self.downsample_factor = 4  # Downsample by factor of 4 to go from 1280x1024 to 320x256
        self.max_width = 512  # Maximum width after downsampling
        self.max_height = 512  # Maximum height after downsampling

    def downsample_input(self, x):
        """Downsample input tensor if it exceeds max dimensions"""
        if x.shape[-2] > self.max_height or x.shape[-1] > self.max_width:
            # Calculate new dimensions
            new_h = x.shape[-2] // self.downsample_factor
            new_w = x.shape[-1] // self.downsample_factor
            
            # Handle different input dimensions
            if len(x.shape) == 2:  # 2D array (H, W)
                return x[::self.downsample_factor, ::self.downsample_factor]
            elif len(x.shape) == 3:  # 3D array (C, H, W) or (H, W, C)
                if x.shape[0] < x.shape[1]:  # (C, H, W)
                    return x[:, ::self.downsample_factor, ::self.downsample_factor]
                else:  # (H, W, C)
                    return x[::self.downsample_factor, ::self.downsample_factor, :]
            elif len(x.shape) == 4:  # 4D array (B, C, H, W)
                return x[:, :, ::self.downsample_factor, ::self.downsample_factor]
        return x

    def upsample_output(self, x, target_size):
        """Upsample output tensor to match target size"""
        if x.shape[-2] != target_size[-2] or x.shape[-1] != target_size[-1]:
            return self.upsample(x)
        return x

    def anomalySource(self, x0):
        p = np.random.rand()
        textural_border = 0.5
        if p < textural_border:
            img = self.backgrounds.__getitem__(
                np.random.randint(0, len(self.backgrounds))
            )["img"]
        else:
            img = self.structureSource(x0)
        return img

    def structureSource(self, img: np.ndarray) -> np.ndarray:
        img = img.transpose((1, 2, 0))
        img = (img / 2) + 0.5
        img = (img * 255).astype(np.uint8)
        structure_source_img = self.backgrounds.randAugmenter()(image=img)

        # Calculate grid dimensions based on actual image size
        h, w = structure_source_img.shape[:2]
        grid_h = h // self.structure_grid_size[0]
        grid_w = w // self.structure_grid_size[1]

         # Ensure grid dimensions are valid
        if grid_h * self.structure_grid_size[0] != h or grid_w * self.structure_grid_size[1] != w:
            # Adjust grid size to match image dimensions
            self.structure_grid_size = [h // grid_h, w // grid_w]

        structure_source_img = rearrange(
            tensor=structure_source_img,
            pattern="(h gh) (w gw) c -> (h w) gw gh c",
            gw=grid_w,
            gh=grid_h,
        )
        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)
        structure_source_img = (
            rearrange(
                tensor=structure_source_img[disordered_idx],
                pattern="(h w) gw gh c -> (h gh) (w gw) c",
                h=self.structure_grid_size[0],
                w=self.structure_grid_size[1],
            ).astype(np.float32)
            / 255
        )
        structure_source_img = structure_source_img.transpose((2, 0, 1))
        structure_source_img = (structure_source_img - 0.5) * 2
        return structure_source_img

    def getPerlinNoise(self, x0, depth=False):
        perlin_scale = 6
        min_perlin_scale = 0
        perlin_scalex = 2 ** (
            torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0]
        )
        perlin_scaley = 2 ** (
            torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0]
        )
        noiseMap = MathUtils.rand_perlin_2d(
            (x0.shape[1], x0.shape[2]), (perlin_scalex, perlin_scaley)
        )
        noiseMap = self.rot(image=np.array(noiseMap))
        if depth:
            perlin_noise = noiseMap

            perlin_thr = np.where(
                np.abs(perlin_noise) > 0.5,
                np.ones_like(perlin_noise),
                np.zeros_like(perlin_noise),
            )
            perlin_thr = np.expand_dims(perlin_thr, axis=2)

            norm_perlin = np.where(
                np.abs(perlin_noise) > 0.5, perlin_noise, np.zeros_like(perlin_noise)
            )
            return norm_perlin, perlin_thr, perlin_noise, 0.5
        return noiseMap

    def generateNoiseBatch(self, x0, t, idx=None, fg_mask=None):
        if self.d and self.rgb:
            d = x0[3:, :, :]
            x0 = x0[:3, :, :]
        elif self.d:
            d = x0

        if self.d:
            plane_mask = (
                np.ones((self.image_width, self.image_height))
                if fg_mask is None
                else fg_mask.detach().cpu().numpy()
            )
            plane_mask = plane_mask[:, :, None]
            image = d.detach().cpu().numpy().transpose((1, 2, 0))
            plane_mask = plane_mask.transpose((1, 0, 2))
            perlin_norm, perlin_thr, perlin_noise, p_thr = self.getPerlinNoise(x0, True)
            perlin_thr = perlin_thr * plane_mask

            zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))

            msk = (perlin_thr).astype(np.float32)
            msk[:, :, 0] = msk[:, :, 0] * (1.0 - zero_mask[:, :, 0])

            perlin_norm_pos = np.where(
                perlin_norm > 0, perlin_norm - p_thr, np.zeros_like(perlin_norm)
            )
            perlin_norm_pos = perlin_norm_pos / (
                np.max(perlin_norm_pos) + 1e-12
            )  # from 0 to 1
            perlin_norm_neg = np.where(
                perlin_norm < 0, perlin_norm + p_thr, np.zeros_like(perlin_norm)
            )
            perlin_norm_neg = perlin_norm_neg / (
                np.abs(np.min(perlin_norm_neg)) + 1e-12
            )  # from 0 to -1
            perlin_norm_mod = (
                perlin_norm_pos + perlin_norm_neg
            )  # hills start from 0 and are of magnitude -1 to 1

            noiseMap = torch.FloatTensor(perlin_noise).squeeze()
            prob = np.random.rand()
            if prob < 1 / 4:
                threshold_corr = np.max(noiseMap.detach().cpu().numpy())
            elif prob < 1 / 2:
                threshold_corr = 0.5 + 0.5 * np.random.rand()
            elif prob < 3 / 4:
                threshold_corr = 0.5
            else:
                threshold_corr = 0.5 - 0.5 * np.random.rand()
            noiseMapCorruptedDepth = np.where(
                np.abs(noiseMap) > threshold_corr,
                np.ones_like(noiseMap),
                np.zeros_like(noiseMap),
            )
            noiseMapCorruptedDepth = np.expand_dims(noiseMapCorruptedDepth, axis=2)
            noiseMapCorruptedDepth = noiseMapCorruptedDepth * plane_mask
            noiseMapCorruptedDepth = (noiseMapCorruptedDepth).astype(np.float32)
            noiseMapCorruptedDepth[:, :, 0] = noiseMapCorruptedDepth[:, :, 0] * (
                1.0 - zero_mask[:, :, 0]
            )
            noiseMapCorruptedDepth = torch.FloatTensor(noiseMapCorruptedDepth).cuda()

            thr = self.lmax
            noiseMapDepth = perlin_norm_mod * thr
            noiseMapDepth = noiseMapDepth * (1.0 - zero_mask[:, :, 0])
            noiseMapDepth = torch.FloatTensor(noiseMapDepth).cuda()
            noiseMapDepth = torch.tile(
                noiseMapDepth.unsqueeze(0), (d.shape[0], 1, 1)
            ).cuda()

        if self.rgb:
            #print(x0.shape)
            noiseMap = self.getPerlinNoise(x0, False)
            #print(np.zeros_like(noiseMap).shape)
            if self.use_pretrained_masks:
                foregroundMask = self.maskDataset.__getitem__(idx)
                #print(foregroundMask.shape)
                if self.is_texture:
                    foregroundMask = np.ones_like(foregroundMask)
            else:
                foregroundMask = np.ones((noiseMap.shape[-2], noiseMap.shape[-1]))

            img = self.anomalySource(x0.cpu().numpy())
            img = torch.FloatTensor(img)

            if self.d:
                noiseMapCorrupted = noiseMapCorruptedDepth.squeeze().cuda()
                noiseMap = torch.FloatTensor(msk).squeeze()
            else:
                prob = np.random.rand()
                if prob < 1 / 4:
                    threshold_corr = np.max(noiseMap)
                elif prob < 1 / 2:
                    threshold_corr = 0.5 + (1 - 0.5) * np.random.rand()
                elif prob < 3 / 4:
                    threshold_corr = 0.5
                else:
                    threshold_corr = 0.5 - (1 - 0.5) * np.random.rand()
                noiseMapCorrupted = np.where(
                    np.abs(noiseMap) > threshold_corr,
                    np.ones_like(noiseMap),
                    np.zeros_like(noiseMap),
                )

                noiseMap = np.where(
                    np.abs(noiseMap) > 0.5,
                    np.ones_like(noiseMap),
                    np.zeros_like(noiseMap),
                )
                noiseMap = noiseMap * foregroundMask
                noiseMapCorrupted = noiseMapCorrupted * foregroundMask

                noiseMapCorrupted = torch.FloatTensor(noiseMapCorrupted).cuda()
                noiseMap = torch.FloatTensor(noiseMap)

            # Construct the final anomaly
            noiseMapImg = noiseMap.unsqueeze(0)
            noiseMapImg = torch.tile(noiseMapImg, (x0.shape[0], 1, 1)).cuda()
            img = img.cuda()
            if img.shape != noiseMapImg.shape:
                    if img.shape == (noiseMapImg.shape[0], noiseMapImg.shape[2], noiseMapImg.shape[1]):
                        img = img.transpose(-2, -1)  # Swap last two dimensions
                    else:
                        raise ValueError(f"Incompatible shapes: {noiseMapImg.shape} vs {img.shape}")
            anomaly_img = noiseMapImg * img

        if self.d and self.rgb:
            msk = torch.FloatTensor(msk).squeeze().unsqueeze(0)
            noiseMapCorruptedDepth = noiseMapCorruptedDepth.squeeze().unsqueeze(0)
            full_noise_map = msk.unsqueeze(0)
            full_corr_noise_map = noiseMapCorruptedDepth.unsqueeze(0)
            noiseMapDepth = noiseMapDepth * msk.cuda()
            full_anomaly_img = torch.cat((anomaly_img, noiseMapDepth), dim=0)

            return (
                full_anomaly_img,
                full_noise_map,
                full_corr_noise_map,
            )
        elif self.d:
            msk = torch.FloatTensor(msk).squeeze().unsqueeze(0)
            noiseMapCorruptedDepth = noiseMapCorruptedDepth.squeeze().unsqueeze(0)
            noiseMapDepth = noiseMapDepth * msk.cuda()
            return (noiseMapDepth, msk, noiseMapCorruptedDepth)
        else:
            return (anomaly_img, noiseMap, noiseMapCorrupted)

    def returnNoise(self, x0, t, idx=None, plane_mask=None):
        b, c, x, y = x0.shape
        eps = torch.zeros((b, c, x, y))
        mask = torch.zeros((b, 1, x, y))
        corr_mask = torch.zeros((b, 1, x, y))
        for bi in range(x0.shape[0]):
            ti = t[bi]
            e, n, nc = self.generateNoiseBatch(
                x0[bi, :, :, :],
                ti,
                idx=idx[bi],
                fg_mask=None if plane_mask is None else plane_mask[bi, :, :],
            )
            eps[bi, :, :, :] = e
            mask[bi, :, :, :] = n
            corr_mask[bi, :, :, :] = nc
        return eps.cuda(), mask.cuda(), corr_mask.cuda()
