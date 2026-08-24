# GPU port of the IEBCS DVS sensor model (IEBCS/src/dvs_sensor.py).
# Same API (initCamera / init_image / update) and same ICNS event semantics,
# with all per-pixel state held as torch tensors on the GPU. update() returns
# a regular EventBuffer, so downstream consumers are unchanged.
#
# Only the NOISE_FREQ noise model is implemented (the one train.py uses).
# Per-event randomness (threshold renewal, latency jitter) uses torch's RNG,
# so event streams match the numpy version statistically, not bit-for-bit.
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "IEBCS", "src"))
from event_buffer import EventBuffer

# Sentinel for "not in refractory". The numpy version uses uint64 max; torch has
# no uint64, so use a value comfortably larger than any simulated timestamp.
REF_NONE = 2 ** 62


class TorchDvsSensor:
    """Drop-in replacement for DvsSensor with GPU-resident state."""

    def __init__(self, name, device=None):
        self.name = name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def initCamera(self, x, y, lat, jit, ref, tau, th_pos, th_neg, th_noise, bgnp, bgnn):
        d = self.device
        self.shape = (y, x)  # (H, W), matching the numpy version's swap
        H, W = self.shape
        self.m_th_pos = th_pos
        self.m_th_neg = -th_neg
        self.m_th_noise = th_noise
        self.m_latency = lat
        self.tau = tau
        self.m_jitter = jit
        self.m_bgn_pos_per = int(1e6 / bgnp)
        self.m_bgn_neg_per = int(1e6 / bgnn)
        self.ref = int(ref)

        self.last_v = torch.zeros((H, W), dtype=torch.float32, device=d)
        self.cur_v = torch.zeros((H, W), dtype=torch.float32, device=d)
        self.tau_p = torch.zeros((H, W), dtype=torch.float32, device=d)
        self.cur_ref = torch.full((H, W), REF_NONE, dtype=torch.int64, device=d)
        self.time_px = torch.zeros((H, W), dtype=torch.int64, device=d)
        self.bgn_pos_next = torch.randint(0, self.m_bgn_pos_per, (H, W), dtype=torch.int64, device=d)
        self.bgn_neg_next = torch.randint(0, self.m_bgn_neg_per, (H, W), dtype=torch.int64, device=d)
        self.cur_th_pos = torch.clamp(
            torch.normal(self.m_th_pos, th_noise, size=(H, W), device=d), 0, 1000)
        self.cur_th_neg = torch.clamp(
            torch.normal(self.m_th_neg, th_noise, size=(H, W), device=d), -1000, 0)
        self.time = 0

        yy, xx = torch.meshgrid(torch.arange(H, device=d), torch.arange(W, device=d), indexing="ij")
        self._yy, self._xx = yy, xx
        self._ref_none = torch.tensor(REF_NONE, dtype=torch.int64, device=d)

    def init_image(self, img):
        t = torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float32, device=self.device)
        self.last_v = torch.log(t + 1)
        self.cur_v = self.last_v.clone()
        self.tau_p = self.tau * 1e3 / (t + 1)
        self.time_px.zero_()
        self.time = 0

    def _end_refractory(self, mask, img_l):
        """Reset pixels whose refractory period expired: sample the voltage at the
        reset time, end refractory, and rebase the reference voltage."""
        delta = (self.cur_ref - self.time_px).to(torch.float32)
        v = self.cur_v + (img_l - self.cur_v) * (1 - torch.exp(-delta / self.tau_p))
        self.last_v = torch.where(mask, v, self.last_v)
        self.time_px = torch.where(mask, self.cur_ref, self.time_px)
        self.cur_ref = torch.where(mask, self._ref_none, self.cur_ref)
        self.cur_v = torch.where(mask, self.last_v, self.cur_v)

    def _fire(self, mask, th_map, m_th_mean, th_lo, th_hi, pol, img_l, ev):
        """Emit events for pixels whose voltage change crossed th_map, renew their
        thresholds and start their refractory periods. Mirrors get_latency_tau."""
        denom = img_l - self.cur_v
        denom = torch.where(denom == 0, torch.full_like(denom, 1e-9), denom)
        amp = (self.last_v + th_map - self.cur_v) / denom
        jit = torch.sqrt(self.m_jitter ** 2 + (self.m_th_noise * self.tau_p / denom) ** 2)
        mean = self.m_latency - self.tau_p * torch.log1p(-amp)
        t_ev = torch.normal(torch.nan_to_num(mean, nan=10000.0), torch.nan_to_num(jit, nan=0.0))
        t_ev = torch.clamp(torch.nan_to_num(t_ev, nan=10000.0), 0, 10000).to(torch.int64)
        ts = self.time_px + t_ev

        ev[0].append(ts[mask])
        ev[1].append(self._yy[mask])
        ev[2].append(self._xx[mask])
        ev[3].append(torch.full((int(mask.sum()),), pol, dtype=torch.int64, device=self.device))

        new_th = torch.clamp(
            torch.normal(m_th_mean, self.m_th_noise, size=th_map.shape, device=self.device),
            th_lo, th_hi)
        self.cur_ref = torch.where(mask, ts + self.ref, self.cur_ref)
        return torch.where(mask, new_th, th_map)

    @torch.no_grad()
    def update(self, img, dt):
        """Advance the sensor by dt microseconds with a new irradiance frame.
        Returns an EventBuffer of the created events, sorted by timestamp."""
        if img.shape[1] != self.shape[1] or img.shape[0] != self.shape[0]:
            print("Error: the size of the image doesn't match with the sensor ")
            return
        d = self.device
        dt = int(dt)
        t_end = self.time + dt

        img_t = torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float32, device=d)
        lit = img_t > 0
        if not bool(lit.any()):
            print("ERROR: update: flux image with only zeros")
            return
        img_l = torch.where(lit, torch.log1p(img_t), img_t)
        self.tau_p = torch.where(lit, self.tau * 1e3 / (img_t + 1), self.tau_p)

        ev = ([], [], [], [])  # ts, y, x, p

        # End refractory periods expiring within this frame. (The numpy version
        # measures this first delta from self.time rather than time_px; they are
        # identical here because time_px == self.time at frame start.)
        self._end_refractory(self.cur_ref < t_end, img_l)

        # Background noise events (NOISE_FREQ model): fixed period per polarity,
        # random phase per pixel. A noise event rebases the pixel like a real one.
        for nxt, per, pol in ((self.bgn_pos_next, self.m_bgn_pos_per, 1),
                              (self.bgn_neg_next, self.m_bgn_neg_per, 0)):
            m = nxt < t_end
            ev[0].append(nxt[m])
            ev[1].append(self._yy[m])
            ev[2].append(self._xx[m])
            ev[3].append(torch.full((int(m.sum()),), pol, dtype=torch.int64, device=d))
            self.time_px = torch.where(m, nxt, self.time_px)
            self.cur_v = torch.where(m, img_l, self.cur_v)
            self.last_v = torch.where(m, img_l, self.last_v)
            nxt += per * m.to(torch.int64)

        # Low-pass-filtered voltage at end of frame; threshold crossings fire.
        delta = (t_end - self.time_px).to(torch.float32)
        target = torch.where(
            lit, self.cur_v + (img_l - self.cur_v) * (1 - torch.exp(-delta / self.tau_p)),
            torch.zeros_like(img_l))
        dif = target - self.last_v
        pos = (dif > self.cur_th_pos) & (self.cur_ref == REF_NONE)
        neg = (dif < self.cur_th_neg) & (self.cur_ref == REF_NONE)

        # A pixel can fire, sit out its refractory period, and fire again within
        # the same frame; iterate until no new crossings appear.
        iters = 0
        while bool(pos.any()) or bool(neg.any()):
            iters += 1
            if iters > 5000:
                print("WARNING: TorchDvsSensor.update: crossing loop did not converge")
                break
            self.cur_th_pos = self._fire(pos, self.cur_th_pos, self.m_th_pos, 0, 1000, 1, img_l, ev)
            self.cur_th_neg = self._fire(neg, self.cur_th_neg, self.m_th_neg, -1000, 0, 0, img_l, ev)

            m_ref = self.cur_ref < t_end
            self._end_refractory(m_ref, img_l)
            # Only pixels that just left refractory can cross again this frame.
            delta = (t_end - self.time_px).to(torch.float32)
            tgt = self.cur_v + (img_l - self.cur_v) * (1 - torch.exp(-delta / self.tau_p))
            dif = torch.where(m_ref, tgt - self.last_v, torch.zeros_like(img_l))
            pos = dif > self.cur_th_pos
            neg = dif < self.cur_th_neg

        # Settle pixel voltages at end of frame and advance time.
        delta = (t_end - self.time_px).to(torch.float32)
        v = self.cur_v + (img_l - self.cur_v) * (1 - torch.exp(-delta / self.tau_p))
        self.cur_v = torch.where(lit, v, self.cur_v)
        self.time = t_end
        self.time_px.fill_(t_end)

        # Gather to an EventBuffer, sorted by timestamp (as pk_end.sort() does).
        ts = torch.cat(ev[0])
        n = int(ts.shape[0])
        pk = EventBuffer(max(n, 1))
        if n > 0:
            order = torch.argsort(ts)
            pk.ts[:n] = ts[order].cpu().numpy().astype(np.uint64)
            pk.y[:n] = torch.cat(ev[1])[order].cpu().numpy().astype(np.uint16)
            pk.x[:n] = torch.cat(ev[2])[order].cpu().numpy().astype(np.uint16)
            pk.p[:n] = torch.cat(ev[3])[order].cpu().numpy().astype(np.uint8)
        pk.i = n
        return pk
