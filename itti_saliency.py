# GPU implementation of the Itti-Koch-Niebur saliency model.
#
#   Itti, Koch & Niebur (1998), "A Model of Saliency-Based Visual Attention for
#   Rapid Scene Analysis", IEEE TPAMI.
#
# Intensity, color-opponency (RG, BY) and orientation (Gabor) channels; dyadic
# Gaussian pyramids; center-surround differences across scales; the N(.)
# normalization operator promoting maps with few strong peaks; conspicuity maps
# summed into a single saliency map. Used as an RGB-based alternative to the
# event-driven spiking attention (train.py --saliency_source itti).
import cv2
import numpy as np
import torch
import torch.nn.functional as F

N_LEVELS = 9            # pyramid sigma = 0..8
CENTERS = (2, 3, 4)
DELTAS = (3, 4)         # surround = center + delta
OUT_LEVEL = 4           # feature maps are combined at this scale


class IttiKochSaliency:
    def __init__(self, device):
        self.device = torch.device(device)
        # 5x5 binomial kernel for pyramid smoothing
        k = np.array([1, 4, 6, 4, 1], dtype=np.float32)
        k = np.outer(k, k) / 256.0
        self.blur_k = torch.tensor(k, device=self.device).view(1, 1, 5, 5)
        # Gabor bank at 0/45/90/135 degrees (real part)
        gabors = [cv2.getGaborKernel((11, 11), sigma=2.8, theta=np.deg2rad(th),
                                     lambd=5.6, gamma=0.5, psi=0).astype(np.float32)
                  for th in (0, 45, 90, 135)]
        gabors = [g - g.mean() for g in gabors]
        self.gabor_k = torch.tensor(np.stack(gabors), device=self.device).unsqueeze(1)

    def _pyramid(self, x):
        levels = [x]
        for _ in range(N_LEVELS - 1):
            x = F.conv2d(x, self.blur_k, padding=2)
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear',
                              align_corners=False, recompute_scale_factor=False)
            levels.append(x)
        return levels

    @staticmethod
    def _cs(pyr_c, pyr_s):
        """Across-scale center-surround difference at the center's resolution."""
        up = F.interpolate(pyr_s, size=pyr_c.shape[-2:], mode='bilinear', align_corners=False)
        return (pyr_c - up).abs()

    @staticmethod
    def _norm(m):
        """Itti's N(.): scale to [0,1], then weight by (1 - mean of local maxima)^2
        so maps with a single strong peak dominate maps with many similar peaks."""
        mn, mx = m.min(), m.max()
        if (mx - mn) < 1e-12:
            return torch.zeros_like(m)
        m = (m - mn) / (mx - mn)
        h, w = m.shape[-2:]
        lm = F.max_pool2d(m, kernel_size=(max(h // 8, 1), max(w // 8, 1)))
        return m * (1.0 - lm.mean()) ** 2

    def _sum_at_out(self, maps, size):
        total = torch.zeros((1, 1) + size, device=self.device)
        for m in maps:
            total += F.interpolate(self._norm(m), size=size, mode='bilinear', align_corners=False)
        return total

    @torch.no_grad()
    def compute(self, rgb):
        """rgb: (H, W, 3) uint8 -> saliency map (H, W) float32 scaled to 0..255."""
        H, W = rgb.shape[:2]
        t = torch.as_tensor(np.ascontiguousarray(rgb), device=self.device).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0)
        r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
        inten = (r + g + b) / 3.0

        # Hue decoupled from intensity; suppressed where intensity is negligible
        mask = inten > 0.1 * inten.max()
        denom = torch.where(mask, inten, torch.ones_like(inten))
        rn, gn, bn = (torch.where(mask, c / denom, torch.zeros_like(c)) for c in (r, g, b))
        R = F.relu(rn - (gn + bn) / 2)
        G = F.relu(gn - (rn + bn) / 2)
        B = F.relu(bn - (rn + gn) / 2)
        Y = F.relu((rn + gn) / 2 - (rn - gn).abs() / 2 - bn)

        pI = self._pyramid(inten)
        pR, pG, pB, pY = (self._pyramid(c) for c in (R, G, B, Y))
        pO = [F.conv2d(l, self.gabor_k, padding=5).abs() for l in pI]  # (1,4,h,w) per level

        i_maps, c_maps = [], []
        o_maps = {th: [] for th in range(4)}
        for c in CENTERS:
            for d in DELTAS:
                s = c + d
                i_maps.append(self._cs(pI[c], pI[s]))
                c_maps.append(self._cs(pR[c] - pG[c], pG[s] - pR[s]))
                c_maps.append(self._cs(pB[c] - pY[c], pY[s] - pB[s]))
                for th in range(4):
                    o_maps[th].append(self._cs(pO[c][:, th:th + 1], pO[s][:, th:th + 1]))

        size = pI[OUT_LEVEL].shape[-2:]
        cons_i = self._sum_at_out(i_maps, size)
        cons_c = self._sum_at_out(c_maps, size)
        cons_o = sum(self._norm(self._sum_at_out(o_maps[th], size)) for th in range(4))

        sal = (self._norm(cons_i) + self._norm(cons_c) + self._norm(cons_o)) / 3.0
        sal = F.interpolate(sal, size=(H, W), mode='bilinear', align_corners=False)[0, 0]
        mn, mx = sal.min(), sal.max()
        sal = (sal - mn) / (mx - mn + 1e-12) * 255.0
        return sal.cpu().numpy().astype(np.float32)
