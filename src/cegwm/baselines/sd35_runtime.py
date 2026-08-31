"""Self-contained SD3.5 VAE/FlowMatch runtime used only by external canaries."""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
from PIL import Image


def image_to_tensor(image: Image.Image, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    array=np.asarray(image.convert("RGB").resize((512,512),Image.Resampling.BICUBIC),dtype=np.float32)/255
    return (torch.from_numpy(array).permute(2,0,1).unsqueeze(0)*2-1).to(device=device,dtype=dtype)


class InversionStableDiffusion3PipelineMixin:
    def get_image_latents(self, image: torch.Tensor, *, sample: bool=False) -> torch.Tensor:
        with torch.inference_mode():
            dist=self.vae.encode(image).latent_dist; value=dist.sample() if sample else dist.mode()
            return (value-float(getattr(self.vae.config,"shift_factor",0) or 0))*float(getattr(self.vae.config,"scaling_factor",1) or 1)
    def _conditioning(self, prompt: str, guidance: float) -> tuple[torch.Tensor,torch.Tensor,bool]:
        cfg=guidance>1
        p,n,pp,npool=self.encode_prompt(prompt=prompt,prompt_2=None,prompt_3=None,negative_prompt="",device=self._execution_device,do_classifier_free_guidance=cfg)
        return (torch.cat([n,p]),torch.cat([npool,pp]),True) if cfg else (p,pp,False)
    def _velocity(self, latents: torch.Tensor,timestep: torch.Tensor,embeds: torch.Tensor,pooled: torch.Tensor,guidance: float,cfg: bool) -> torch.Tensor:
        model=torch.cat([latents]*2) if cfg else latents; velocity=self.transformer(hidden_states=model,timestep=timestep.expand(model.shape[0]),encoder_hidden_states=embeds,pooled_projections=pooled,joint_attention_kwargs=getattr(self,"_joint_attention_kwargs",None),return_dict=False)[0]
        return velocity.chunk(2)[0]+guidance*(velocity.chunk(2)[1]-velocity.chunk(2)[0]) if cfg else velocity
    def _schedule(self, steps: int, shape: tuple[int,...]) -> tuple[Any,Any]:
        cfg=self.scheduler.config; kw={}
        if getattr(cfg,"use_dynamic_shifting",False) or (hasattr(cfg,"get") and cfg.get("use_dynamic_shifting",False)):
            get=lambda k,d: cfg.get(k,d) if hasattr(cfg,"get") else getattr(cfg,k,d); seq=(shape[-2]//self.transformer.config.patch_size)*(shape[-1]//self.transformer.config.patch_size); kw["mu"]=seq*((get("max_shift",1.16)-get("base_shift",.5))/(get("max_image_seq_len",4096)-get("base_image_seq_len",256)))+get("base_shift",.5)-get("base_image_seq_len",256)*((get("max_shift",1.16)-get("base_shift",.5))/(get("max_image_seq_len",4096)-get("base_image_seq_len",256)))
        if getattr(cfg,"stochastic_sampling",False) or (hasattr(cfg,"get") and cfg.get("stochastic_sampling",False)): raise RuntimeError("external baseline requires deterministic FlowMatch Euler schedule")
        self.scheduler.set_timesteps(steps,device=self._execution_device,**kw); ts, ss = self.scheduler.timesteps,self.scheduler.sigmas
        if len(ts) != steps or len(ss) != steps + 1: raise RuntimeError("FlowMatch schedule length violates SD3.5 Euler contract")
        return ts, ss
    def denoise_segment(self, latents: torch.Tensor, *, prompt: str,guidance: float,steps: int,start: int,end: int) -> torch.Tensor:
        ts,ss=self._schedule(steps,tuple(latents.shape)); embeds,pooled,cfg=self._conditioning(prompt,guidance); current=latents.clone()
        with torch.inference_mode():
            for i in range(start,end): current=(current.float()+(ss[i+1].float()-ss[i].float())*self._velocity(current,ts[i],embeds,pooled,guidance,cfg)).to(latents.dtype)
        return current
    def invert_flow_matching_latent(self,latents: torch.Tensor,*,prompt: str="",num_inference_steps: int=20,guidance_scale: float=4.5) -> torch.Tensor:
        ts,ss=self._schedule(num_inference_steps,tuple(latents.shape)); embeds,pooled,cfg=self._conditioning(prompt,guidance_scale); current=latents.clone()
        with torch.inference_mode():
            for i in range(num_inference_steps-1,-1,-1): current=(current.float()+(ss[i].float()-ss[i+1].float())*self._velocity(current,ts[i],embeds,pooled,guidance_scale,cfg)).to(latents.dtype)
        return current
    def invert_to_edit_timestep(self,latents: torch.Tensor,*,prompt: str="",num_inference_steps: int=20,edit_fraction: float=.2,guidance_scale: float=4.5) -> torch.Tensor:
        edit=int(edit_fraction*num_inference_steps); index=num_inference_steps-edit; ts,ss=self._schedule(num_inference_steps,tuple(latents.shape)); embeds,pooled,cfg=self._conditioning("",1.0); current=latents.clone()
        with torch.inference_mode():
            for i in range(num_inference_steps-1,index-1,-1): current=(current.float()+(ss[i].float()-ss[i+1].float())*self._velocity(current,ts[i],embeds,pooled,1.0,cfg)).to(latents.dtype)
        return current
    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        with torch.inference_mode(): return self.image_processor.postprocess(self.vae.decode(latents/self.vae.config.scaling_factor+self.vae.config.shift_factor,return_dict=False)[0],output_type="pil")[0]


def load_sd3_pipeline(model_id: str, revision: str, *, token: str) -> Any:
    from diffusers import StableDiffusion3Pipeline
    cls=type("BaselineInversionSD3",(InversionStableDiffusion3PipelineMixin,StableDiffusion3Pipeline),{})
    pipe=cls.from_pretrained(model_id,revision=revision,torch_dtype=torch.float16,token=token).to("cuda"); pipe.transformer.eval(); pipe.vae.eval(); pipe.set_progress_bar_config(disable=True); return pipe
