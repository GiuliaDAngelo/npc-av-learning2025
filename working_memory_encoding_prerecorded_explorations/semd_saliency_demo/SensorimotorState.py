from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
from helpers import *
from attention import *

class SensorimotorState:
    def __init__(self,
                 dt_us: int,
                 net_attention,
                 device: torch.device,
                 att_config, rotate_in_N_steps, foveation_radius = 38, downsample_tile = 8, semd_instance = None):
        self.dt_us = dt_us
        self.net_attention = net_attention
        self.device = device
        self.att_config = att_config
        self.foveation_radius = foveation_radius
        self.downsample_tile = downsample_tile
        self.semd_instance = semd_instance
        self.away_side_sign = +1
        self.rotate_in_N_steps = rotate_in_N_steps

        self.step_idx = 0
        self.t_us = 0
        self.rgb = None
        self.events_img = None
        self.foveated_events = None
        self.saliency = None
        self.fixation_xy = (0.0, 0.0)
        self.rel_x = 0.0
        self.rel_y = 0.0
        self.most_salient_point = None
        self.vx = None
        self.vy = None
        self.speed = None
        self.semd_vis = None
        self.abs_pitch = 0.0
        self.abs_roll = 0.0
        self.d_pitch = 0.0
        self.d_roll = 0.0
        self.rot_axis_img = None
        self.rot_towards_mask = None
        self.rot_away_mask = None
        self.modulated_saliency = None
        self.inhibited_return_mask = None
        

    def update(
        self,
        step_idx: int,
        events: np.ndarray,
        ev_img: np.ndarray,
        fixation_xy: Tuple[float, float], # current position of eye fixation
        abs_pitch: float, # deg
        abs_roll: float, # deg
        divider_axis_img        
    ):
        # sensory input fields
        self.step_idx = step_idx
        self.t_us = step_idx*self.dt_us
        self.events = events 
        self.ev_img = ev_img
        self.fixation_xy = (float(fixation_xy[0]), float(fixation_xy[1]))
        
        self.d_pitch = self.abs_pitch - abs_pitch # calculate relative change in rotation from last step
        self.abs_pitch = abs_pitch # update absolute rotation to current value
        self.d_roll = self.abs_roll - abs_roll
        self.abs_roll = abs_roll
        self.divider_axis_img = divider_axis_img

        # evaluate attention 
        self.saliency_col, self.saliency, _= saliency_from_events(self.ev_img, self.net_attention, self.device, self.att_config)
        self.most_salient_point = np.unravel_index(np.argmax(self.saliency), self.saliency.shape)

        # modulate attention with proprioception rotation slope (towards/away from us)
        if self.divider_axis_img is not None:
            W_axis = axis_sloped_weight(self.ev_img.shape[0], self.ev_img.shape[1], divider_axis_img= divider_axis_img, 
                                        center_xy=((self.ev_img.shape[1] - 1) / 2.0, (self.ev_img.shape[0] - 1) / 2.0),          
                                        towards_side_sign=-1,        # flip to -1 if sides are swapped
                                        boost=8, suppress=0.9, gamma=1.0
            )
            self.modulated_saliency = self.saliency * W_axis
        else:
            self.modulated_saliency = self.saliency

        # calculate SEMD 
        directions, self.angles, self.motion_mag, self.motion_conf, self.motion_valid_positions, self.semd_vis = run_SEMD(self.semd_instance, self.events, HEIGHT=self.ev_img.shape[0], WIDTH = self.ev_img.shape[1])
        self.vx, self.vy = directions
        self.angles = pad_center(self.angles, ev_img.shape[0],ev_img.shape[1], pad_value=0) # pad to original size (only square was used to spare comp. resources)
        # get expected directions from proprioception
        if self.step_idx > 1:
            self.expected_angles_proprioception = infere_angles_from_proprioception(self.ev_img, self.divider_axis_img)
            
            #evaluate surprise in actual directions from semd
            self.surprise, self.cos_sim, valid = angle_cosine_similarity_heatmap(self.angles, self.expected_angles_proprioception,
                block=12, min_valid_frac=1.0, clip_pctl=99.0, gamma=1.0, bg_white=True, cmap=cv2.COLORMAP_JET,
            )   

            # modulate saliency with surprise in motion directionality 
            self.modulated_saliency, _ = modulate_saliency_with_surprise(self.modulated_saliency, self.cos_sim, valid=valid, mode="cos_sim", alpha=4.0)                                                    
      
        # choose next fixation 
        sel = np.isfinite(self.modulated_saliency) & (self.saliency > 0)  
        Ssel = np.where(sel, self.modulated_saliency, -np.inf)
        y, x = np.unravel_index(np.argmax(Ssel), Ssel.shape)

        self.modulated_saliency_peak = (x,y)
        H, W = self.saliency.shape[:2]
        rr, cc = np.ogrid[:H, :W]
        r_inhibit = 30  # size of circle for inhibition in pixels

        if self.inhibited_return_mask is None:
            self.inhibited_return_mask = np.ones_like(self.saliency)

      

        if self.step_idx % self.rotate_in_N_steps == 0:
            self.next_fixation_xy = (x,y)
            # inhibit circle around the current new fixation
            self.inhibited_return_mask = np.ones_like(self.saliency, dtype=np.float32)
            circle = (rr - y)**2 + (cc - x)**2 <= r_inhibit**2
            self.inhibited_return_mask[circle] = 0.0
            self.saliency_inhibited = self.inhibited_return_mask * self.saliency
        else:
            # get inhibited saliency map
            self.saliency_inhibited = self.inhibited_return_mask * self.saliency
            (y, x) = np.unravel_index(np.argmax(self.saliency_inhibited), self.saliency.shape)
            self.next_fixation_xy = (x,y)

            circle = (rr - self.next_fixation_xy[1])**2 + (cc - self.next_fixation_xy[0])**2 <= r_inhibit**2

            self.inhibited_return_mask[circle] = 0.0


        self.rel_x = self.fixation_xy[0] - self.next_fixation_xy[0]
        self.rel_y = self.fixation_xy[1] - self.next_fixation_xy[1]

        # transform input to foveated 
        self.foveated, self.foveated_color = make_foveated_events_eccentricity(self.ev_img, self.next_fixation_xy[0], self.next_fixation_xy[1], self.foveation_radius, n_rings=6, max_tile=self.downsample_tile)
        #self.foveated =  make_foveated_events_binary(self.ev_img, self.next_fixation_xy[0], self.next_fixation_xy[1], self.foveation_radius, self.downsample_tile)

        # prepare colormapped visualisations
        # illustrate speed with rotation axis
        self.speed_vis = velocity_magnitude_heatmap(
                self.vx, self.vy,
                block=8, clip_pctl=95.0, gamma=0.7, bg_white=True,
                arrow_len_px=110, arrow_thickness=3, arrow_color=(0,0,0),
                arrow_offset_px=(160, 0), weighted_global=True,
                divider_axis_img=self.divider_axis_img, 
                away_side_sign=self.away_side_sign,
                divider_center_xy=None,  
            )
        
        self.direction_vis = velocity_orientation_heatmap(
            self.angles,              # HxW, 0..360
            mask=None,                # HxW bool (e.g., active events)
            block=6,
            bg_white=True,
            sat=255,
            val=255,
            min_valid_frac=0.1,      # require some valid pixels in block
            divider_axis_img=self.divider_axis_img,
            )
        


      

        