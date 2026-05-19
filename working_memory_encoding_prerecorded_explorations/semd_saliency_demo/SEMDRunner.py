import numpy as np
from pygenn import GeNNModel, create_neuron_model, init_postsynaptic, init_weight_update
import matplotlib.pyplot as plt
import cv2

# Tuned for a single time window (single velocity), it will spike at maximum when the velocity is matched exactly, but will spike less for both lower and higher velocities. 
# Directions are provided precisely, but the magnitudes are proxy only. Calibration or multiple differently tuned semd would be needed for accurate velocity. 

class SEMDRunner:
    """
    Builds SEMD once, then runs it on many event windows by overwriting SpikeSourceArray spikeTimes.
    Outputs are spike counts in TB/BT/LR/RL populations for the window.
    """
    def __init__(self, H, W, dt_ms=1.0, backend="single_threaded_cpu",
                 semd_code=None, semd_threshold=None, semd_reset=None):
        self.H = int(H)
        self.W = int(W)
        self.dt_ms = float(dt_ms)
        self.backend = backend

        if semd_code is None or semd_threshold is None or semd_reset is None:
            raise ValueError("Pass semd_code/semd_threshold/semd_reset from semdCode (code, threshold, reset).")

        self._semd_code = semd_code
        self._semd_threshold = semd_threshold
        self._semd_reset = semd_reset

        self._build_model()

    def _build_model(self):
        H, W = self.H, self.W

        # Build GeNN model
        self.model = GeNNModel(precision="float", model_name=".SEMD", backend=self.backend)
        self.model.dt = self.dt_ms  # ms

        # Spike source (one neuron per pixel)
        # startSpike/endSpike are indices into the global spikeTimes array
        n_src = H * W
        self.popSource = self.model.add_neuron_population(
            pop_name="stim",
            num_neurons=n_src,
            neuron="SpikeSourceArray",
            params={},
            vars={
                "startSpike": np.zeros(n_src, dtype=np.uint32),
                "endSpike":   np.zeros(n_src, dtype=np.uint32),
            },
        )
        self.popSource.spike_recording_enabled = True

        # SEMD neuron model
        semdModel = create_neuron_model(
            class_name="semd",
            params=["C", "TauM", "Vrest", "Vreset", "Vthresh", "Ioffset", "TauRefrac", "TauSynTrig"],
            vars=[("V", "scalar"), ("RefracTime", "scalar"), ("ISynTrigger", "scalar")],
            derived_params=[
                ("ExpTC", lambda pars, dt: np.exp(-dt / pars["TauM"])),
                ("Rmembrane", lambda pars, dt: pars["TauM"] / pars["C"]),
                ("trigExpDecay", lambda pars, dt: np.exp(-dt / pars["TauSynTrig"])),
                ("trigInit", lambda pars, dt: (pars["TauSynTrig"] * (1.0 - np.exp(-dt / pars["TauSynTrig"]))) * (1.0 / dt)),
            ],
            sim_code=self._semd_code,
            threshold_condition_code=self._semd_threshold,
            reset_code=self._semd_reset,
            additional_input_vars=[("ISynFac", "scalar", 0.0)],
        )

        semdParams = {
            "C": 0.25,
            "TauM": 10.0,
            "Vrest": 0.0,
            "Vreset": 0.0,
            "Vthresh": 2.0,
            "Ioffset": 0.0,
            "TauRefrac": 0.0,
            "TauSynTrig": 10.0,
        }
        semdVars = {"V": 0.0, "RefracTime": 0.0, "ISynTrigger": 0.0}

        # Output populations
        self.popLR = self.model.add_neuron_population(
            pop_name="LR", num_neurons=H * (W - 1), neuron=semdModel, params=semdParams, vars=semdVars
        )
        self.popRL = self.model.add_neuron_population(
            pop_name="RL", num_neurons=H * (W - 1), neuron=semdModel, params=semdParams, vars=semdVars
        )
        self.popTB = self.model.add_neuron_population(
            pop_name="TB", num_neurons=(H - 1) * W, neuron=semdModel, params=semdParams, vars=semdVars
        )
        self.popBT = self.model.add_neuron_population(
            pop_name="BT", num_neurons=(H - 1) * W, neuron=semdModel, params=semdParams, vars=semdVars
        )

        self.popLR.spike_recording_enabled = True
        self.popRL.spike_recording_enabled = True
        self.popTB.spike_recording_enabled = True
        self.popBT.spike_recording_enabled = True

        # Build connectivity indices
        pixel = np.arange(H * W, dtype=np.uint32).reshape(H, W)

        # Horizontal (LR/RL)
        synHorFac = pixel[:, :-1].flatten()  # x
        synHorTri = pixel[:, 1:].flatten()   # x+1
        synHor = np.arange(H * (W - 1), dtype=np.uint32)

        # Vertical (TB/BT)
        synVerFac = pixel[:-1, :].T.flatten()  # y
        synVerTri = pixel[1:, :].T.flatten()   # y+1
        synVer = np.arange((H - 1) * W, dtype=np.uint32).reshape((H - 1), W).T.flatten()

        # Synapses: facilitatory (ExpCurr) + trigger (DeltaCurr)
        def add_pair(name_fac, name_tri, target_pop, src_ids_fac, src_ids_tri, tgt_ids):
            fac = self.model.add_synapse_population(
                pop_name=name_fac, matrix_type="SPARSE",
                source=self.popSource, target=target_pop,
                postsynaptic_init=init_postsynaptic("ExpCurr", {"tau": 5.0}),
                weight_update_init=init_weight_update("StaticPulse", {}, {"g": np.ones_like(tgt_ids)}),
            )
            fac.post_target_var = "ISynFac"
            fac.set_sparse_connections(src_ids_fac, tgt_ids)

            tri = self.model.add_synapse_population(
                pop_name=name_tri, matrix_type="SPARSE",
                source=self.popSource, target=target_pop,
                postsynaptic_init=init_postsynaptic("DeltaCurr"),
                weight_update_init=init_weight_update("StaticPulse", {}, {"g": np.ones_like(tgt_ids) * 20}),
            )
            tri.set_sparse_connections(src_ids_tri, tgt_ids)

        add_pair("facLR", "triLR", self.popLR, synHorFac, synHorTri, synHor)
        add_pair("facRL", "triRL", self.popRL, synHorTri, synHorFac, synHor)
        add_pair("facTB", "triTB", self.popTB, synVerFac, synVerTri, synVer)
        add_pair("facBT", "triBT", self.popBT, synVerTri, synVerFac, synVer)

        # Build code once
        self.model.build()

    @staticmethod
    def _rec_count(rec):
        # PyGeNN commonly returns (times, ids) as a tuple
        if isinstance(rec, (tuple, list)) and len(rec) == 2:
            return len(rec[0])
        return len(rec)

    @staticmethod
    def _make_start_end(sourceGenn):
        # sourceGenn: list length Npix; each entry is 1D float times in ms (float32/float64)
        lengths = np.fromiter((len(s) for s in sourceGenn), count=len(sourceGenn), dtype=np.uint32)
        end = np.cumsum(lengths, dtype=np.uint32)
        start = np.empty_like(end)
        start[0] = 0
        start[1:] = end[:-1]

        if end[-1] == 0:
            spikes = np.empty((0,), dtype=np.float32)
        else:
            spikes = np.concatenate(sourceGenn).astype(np.float32, copy=False)

        return start, end, spikes

    def run_window(self, sourceGenn, sim_steps):
        start, end, spikeTimes = self._make_start_end(sourceGenn)
        max_t = float(spikeTimes.max()) if spikeTimes.size else 0.0
        sim_steps = int(np.ceil(max_t / self.dt_ms)) + 1
        sim_steps = max(sim_steps, 1)

        # DEBUG: prove you have temporal structure
        total = int(end[-1]) if end.size else 0
        if total:
            #t("[SEMD] spikes total:", total,
            #    "min/max step:", int(spikeTimes.min()), int(spikeTimes.max()))
            # if max is large, sim_steps must be >= max+1
            sim_steps = max(int(sim_steps), int(spikeTimes.max()) + 1)
        else:
            #print("[SEMD] spikes total: 0")
            sim_steps = max(int(sim_steps), 1)

        self.popSource.vars["startSpike"].set_init_values(start)
        self.popSource.vars["endSpike"].set_init_values(end)
        self.popSource.extra_global_params["spikeTimes"].set_init_values(spikeTimes)

        self.model.load(num_recording_timesteps=sim_steps)

        while self.model.timestep < sim_steps:
            self.model.step_time()

        self.model.pull_recording_buffers_from_device()
        def rec_n(rec):
            return len(rec[0]) if isinstance(rec, (tuple, list)) and len(rec) == 2 else len(rec)

        src_rec = self.popSource.spike_recording_data[0]
        tb_rec  = self.popTB.spike_recording_data[0]
        lr_rec  = self.popLR.spike_recording_data[0]

        #print("[SEMD] injected total:", int(end[-1]) if end.size else 0,
         #   "SRC recorded:", rec_n(src_rec),
         #   "TB recorded:", rec_n(tb_rec),
          #  "LR recorded:", rec_n(lr_rec))

        # ---- Extract full spatial maps ----
        def pop_to_map(pop, shape):
            rec = pop.spike_recording_data[0]

            # Expect (times, ids)
            if isinstance(rec, (tuple, list)) and len(rec) == 2:
                ids = rec[1].astype(np.int64)
                counts = np.bincount(ids, minlength=pop.num_neurons)
            else:
                # fallback
                counts = np.zeros(pop.num_neurons, dtype=np.int32)

            return counts.reshape(shape)

        H, W = self.H, self.W

        LR = pop_to_map(self.popLR, (H, W-1))
        RL = pop_to_map(self.popRL, (H, W-1))
        TB = pop_to_map(self.popTB, (H-1, W))
        BT = pop_to_map(self.popBT, (H-1, W))

        self.model.unload()

        return TB, BT, LR, RL
    
    def decode_motion_vectors(self, TB, BT, LR, RL, pool_ksize = 3):
        # semd values represent edges between pixels, not centers, thus TB, BT and LR, RL have different shapes (H-1,W) vs (H, W-1)
       
        vx_e = LR - RL # +1 or -1 motion
        vy_e = TB - BT
        # edge representation: shapes (H, W-1), (H-1, W)
        H, Wm1 = vx_e.shape
        Hm1, W = vy_e.shape

        # lift values to pixel centers to get (H,W) shaped arrays
        vx = np.zeros((H,W), dtype=np.float32)
        vx[:, 1:-1] = 0.5 * (vx_e[:, :-1] + vx_e[:, 1:])
        vx[:, 0]    = vx_e[:, 0]
        vx[:, -1]   = vx_e[:, -1]

        vy = np.zeros((H,W), dtype=np.float32)
        vy[1:-1, :] = 0.5 * (vy_e[:-1, :] + vy_e[1:, :])
        vy[0, :]    = vy_e[0, :]
        vy[-1, :]   = vy_e[-1, :]
        
        # magnitude proxy (directional dominance)
        mag = np.sqrt(vx*vx + vy*vy)

        # confidence proxy based on opponent balance (uses raw counts, not centered diffs)
        # lift sums to pixel centers too
        hx_e = (LR + RL).astype(np.float32)      # (H, W-1)
        hy_e = (TB + BT).astype(np.float32)      # (H-1, W)

        hx = np.zeros((H, W), dtype=np.float32)
        hx[:, 1:-1] = 0.5 * (hx_e[:, :-1] + hx_e[:, 1:])
        hx[:, 0]    = hx_e[:, 0]
        hx[:, -1]   = hx_e[:, -1]

        hy = np.zeros((H, W), dtype=np.float32)
        hy[1:-1, :] = 0.5 * (hy_e[:-1, :] + hy_e[1:, :])
        hy[0, :]    = hy_e[0, :]
        hy[-1, :]   = hy_e[-1, :]

        # confidence ~ normalized dominance; suppresses cases where both directions fire equally
        eps = 1e-6
        conf = mag / (hx + hy + eps)             # in [0, ~1] typically

        valid = (hx + hy) > 0                    # any motion evidence at all
        
        angles = np.zeros(vx.shape, dtype=np.float32) 
        ang = (np.degrees(np.arctan2(vy, vx)) + 360.0) % 360.0 #rads to degrees

        angles[valid] = ang[valid]
        angles[(valid) & (angles == 0.0)] = 360.0 # 0 means no motion at all, 360 is for motion to the right (0/360 deg)


    
        # --- 3x3 regional averaging (fast) ---
        # Use box filter on vx, vy, mag/conf; then recompute angle from pooled vx/vy.
        if pool_ksize and pool_ksize > 1:
            k = int(pool_ksize)
            vx_p = cv2.boxFilter(vx, ddepth=-1, ksize=(k, k), normalize=True)
            vy_p = cv2.boxFilter(vy, ddepth=-1, ksize=(k, k), normalize=True)
            mag_p = np.sqrt(vx_p*vx_p + vy_p*vy_p)
            conf_p = cv2.boxFilter(conf, ddepth=-1, ksize=(k, k), normalize=True)
            valid_p = cv2.boxFilter(valid.astype(np.float32), ddepth=-1, ksize=(k, k), normalize=True) > 0.0

            ang_p = (np.degrees(np.arctan2(vy_p, vx_p)) + 360.0) % 360.0

            return (vx_p, vy_p), ang_p, mag_p, conf_p, valid_p

        return (vx, vy), angles, mag, conf, valid

    
def events_to_sourceGenn(xs, ys, ts_us, H, W):
    xs = np.asarray(xs, dtype=np.int32)
    ys = np.asarray(ys, dtype=np.int32)
    ts_us = np.asarray(ts_us)

    if ts_us.size == 0:
        return [np.empty((0,), dtype=np.float32) for _ in range(H * W)]

    # If ts_us is constant, you will get all zeros no matter what:
    # (fix upstream, or synthesize per-event times)
    t0 = ts_us.min()
    t_ms = (ts_us - t0).astype(np.float32) * 1e-3

    pix = (ys.astype(np.int64) * W + xs.astype(np.int64))

    # robust ordering: sort by pixel then by time
    order = np.lexsort((t_ms, pix))
    pix_s = pix[order]
    t_s = t_ms[order]

    source = [np.empty((0,), dtype=np.float32) for _ in range(H * W)]
    uniq, start_idx = np.unique(pix_s, return_index=True)
    end_idx = np.r_[start_idx[1:], len(pix_s)]

    for p, a, b in zip(uniq, start_idx, end_idx):
        source[int(p)] = t_s[a:b]

    return source