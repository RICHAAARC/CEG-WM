"""Saved-RGB mechanism probes. Counterfactual latent scores are NOT blind decisions."""
from __future__ import annotations
import argparse
import csv
from importlib.metadata import version
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

# Same validated VAE-only reader and read-only source verifier as the previous run.
import run_residual as support
from run_residual import (Reader, EXACT, MODEL, IDS, ROLES, ATOL, RTOL, read, write,
    sha, bind, error_data, inspect_source, load_saved_image, replay_comparison)

SUPPORT_SHA = "27945052b071b4a207d674c60cfd79f8d326448a09f5bdb3f54524cc31c1c701"
SHIFTS = ((0,0),(-2,0),(2,0),(0,-2),(0,2),(8,0),(0,8))


def shift_id(dx,dy):
    return f"dx{dx:+d}_dy{dy:+d}"


def rgb_shift(image, dx, dy, periodic=False):
    """Exact uint8 permutation/copy. No RGB interpolation or PIL resampling."""
    import numpy as np
    from PIL import Image
    arr=np.asarray(image)
    shifted=np.roll(arr,(dy,dx),axis=(0,1)).copy()
    if not periodic:
        if dx>0: shifted[:,:dx]=0
        if dx<0: shifted[:,dx:]=0
        if dy>0: shifted[:dy]=0
        if dy<0: shifted[dy:]=0
    return Image.fromarray(shifted)


def project(z):
    """Common real subspace without even-grid Nyquist row/column."""
    import torch
    f=torch.fft.fft2(z.float(),norm="ortho")
    h,w=z.shape[-2:]
    if h%2==0: f[...,h//2,:]=0
    if w%2==0: f[...,:,w//2]=0
    return torch.fft.ifft2(f,norm="ortho").real


def phase_shift(z, dx, dy):
    """Unitary periodic Fourier translation on odd grids or the common subspace."""
    import torch
    h,w=z.shape[-2:]
    f=torch.fft.fft2(z.float(),norm="ortho")
    fy=torch.fft.fftfreq(h,device=z.device)[:,None]
    fx=torch.fft.fftfreq(w,device=z.device)[None,:]
    phase=torch.exp((-2j*math.pi)*(fx*dx+fy*dy))
    shifted=torch.fft.ifft2(f*phase,norm="ortho")
    # Do not silently throw away imaginary Nyquist energy on even grids.
    if float(shifted.imag.norm()) > 2e-5*max(float(shifted.real.norm()),1e-12):
        raise ValueError("Fourier shift left the real common subspace")
    return shifted.real


def padded_phase(z, dx, dy, pad=32):
    import torch.nn.functional as F
    # 64+2*pad+1 is odd: no ambiguous Nyquist mode in the extension grid.
    h,w=z.shape[-2:]
    extended=F.pad(z,(pad,pad+1,pad,pad+1),mode="constant",value=0.)
    moved=phase_shift(extended,dx,dy)
    return moved[...,pad:pad+h,pad:pad+w]


def integer_roll(z, dx, dy, periodic):
    import torch
    result=torch.roll(z,(dy,dx),dims=(-2,-1)).clone()
    if not periodic:
        if dx>0: result[...,:,:dx]=0
        if dx<0: result[...,:,dx:]=0
        if dy>0: result[...,:dy,:]=0
        if dy<0: result[...,dy:,:]=0
    return result


def bilinear_shift(z, dx, dy, periodic):
    ix,iy=math.floor(dx),math.floor(dy)
    ax,ay=dx-ix,dy-iy
    return sum(weight*integer_roll(z,ix+i,iy+j,periodic)
               for i,j,weight in ((0,0,(1-ax)*(1-ay)),(1,0,ax*(1-ay)),
                                  (0,1,(1-ax)*ay),(1,1,ax*ay)) if weight)


def relative(a,b):
    return float((a-b).float().norm()/b.float().norm().clamp_min(1e-12))


def remove_fraction(z):
    import torch
    f=torch.fft.fft2(z.float(),norm="ortho")
    removed=f.clone()
    removed[...,z.shape[-2]//2,:]=0
    removed[...,:,z.shape[-1]//2]=0
    return float(((f-removed).abs()**2).sum()/(f.abs()**2).sum().clamp_min(1e-20))


class HFProbe:
    """Original real-spectrum normalized cosine, full and fixed non-Nyquist masks."""
    def __init__(self, reader, shape, device):
        import torch
        from cegwm.method.hf import _spec
        from cegwm.method.frequency import radial_frequency_mask,reconstruct_frequency_carrier
        spec=_spec(reader.hf)
        h,w=shape[-2:]
        full=torch.from_numpy(radial_frequency_mask(h,w,spec)).to(device)
        common=full.clone()
        common[h//2,:]=False
        common[:,w//2]=False
        self.masks={"full":full,"common":common}
        self.templates={}
        self.labels=("registered",*(f"wrong_{i:02d}" for i in range(16)))
        spectra=[torch.fft.rfft2(reconstruct_frequency_carrier(key,shape,spec,dtype=torch.float32,
                    device=device),norm="ortho").real for key in reader.keys]
        for mode,mask in self.masks.items():
            vectors=[]
            for f in spectra:
                v=f[...,mask].reshape(-1).double()
                v=v-v.mean()
                vectors.append(v/v.norm())
            self.templates[mode]=torch.stack(vectors)

    def score(self,z,mode="common"):
        import torch
        f=torch.fft.rfft2(z.float(),norm="ortho").real
        v=f[...,self.masks[mode]].reshape(-1).double()
        v=v-v.mean()
        norm=v.norm()
        if not torch.isfinite(norm) or float(norm)==0:
            raise RuntimeError("Undefined HF diagnostic cosine")
        values=(self.templates[mode]@(v/norm)).tolist()
        if not all(math.isfinite(x) for x in values):
            raise RuntimeError("Nonfinite diagnostic score")
        return {"scores":dict(zip(self.labels,values)),"registered":values[0],
                "wrong_max":max(values[1:]),"gap":values[0]-max(values[1:])}


def operator_fields(z0,black,z_actual,z_circular,dx,dy):
    """Translate residual relative to encoded black; black baseline stays fixed."""
    r=project(z0-black)
    ax,ay=dx/8,dy/8
    phase=phase_shift(r,ax,ay)
    roundtrip=relative(phase_shift(phase,-ax,-ay),r)
    if roundtrip>2e-5:
        raise RuntimeError("Fourier phase round-trip failed")
    actual=project(z_actual-black)
    circular=project(z_circular-black)
    fields={
        "base":black+r,
        "phase_periodic":black+phase,
        "phase_pad32":black+padded_phase(r,ax,ay,32),
        "phase_pad64":black+padded_phase(r,ax,ay,64),
        "bilinear_periodic":black+bilinear_shift(r,ax,ay,True),
        "bilinear_zero":black+bilinear_shift(r,ax,ay,False),
        "actual_zero":black+actual,
        "actual_circular":black+circular,
        "actual_inverse_periodic":black+phase_shift(circular,-ax,-ay),
        "actual_inverse_pad32":black+padded_phase(actual,-ax,-ay,32),
        "phase_periodic_unsubtracted":phase_shift(project(z0),ax,ay),
    }
    expected=project(fields["phase_pad32"]-black)
    metrics={"phase_roundtrip_relative_l2":roundtrip,
             "phase_periodic_energy_ratio":float(phase.float().norm()/r.float().norm().clamp_min(1e-12)),
             "nyquist_removed_energy_fraction_base_residual":remove_fraction(z0-black),
             "nyquist_removed_energy_fraction_actual_residual":remove_fraction(z_actual-black),
             "extra_relative_l2_full":relative(actual,expected),
             "extra_relative_l2_inner":relative(actual[...,8:56,8:56],expected[...,8:56,8:56]),
             "padding_convergence_relative_l2":relative(fields["phase_pad32"]-black,fields["phase_pad64"]-black)}
    return fields,metrics


def decompose(scores):
    """Signed telescoping score changes, NOT causal variance percentages."""
    result={}
    for metric in ("registered","gap"):
        s={name:value[metric] for name,value in scores.items()}
        result[metric]={
            "total_loss":s["base"]-s["actual_zero"],
            "phase_loss":s["base"]-s["phase_periodic"],
            "boundary_model_delta":s["phase_periodic"]-s["phase_pad32"],
            "extra_vs_padded_phase":s["phase_pad32"]-s["actual_zero"],
            "bilinear_model_delta":s["phase_pad32"]-s["bilinear_zero"],
            "extra_vs_bilinear":s["bilinear_zero"]-s["actual_zero"],
            "periodic_interpolation_delta":s["phase_periodic"]-s["bilinear_periodic"],
            "padding_32_vs_64":s["phase_pad32"]-s["phase_pad64"],
            "rgb_boundary_delta":s["actual_circular"]-s["actual_zero"],
            "black_baseline_model_delta":s["phase_periodic"]-s["phase_periodic_unsubtracted"],
            "inverse_periodic_gain":s["actual_inverse_periodic"]-s["actual_circular"],
            "inverse_pad32_gain":s["actual_inverse_pad32"]-s["actual_zero"],
            "inverse_pad32_shortfall":s["base"]-s["actual_inverse_pad32"],
        }
    return result


def summarize(output):
    rows=[]
    for uid in IDS:
        for role in ROLES:
            for dx,dy in SHIFTS:
                path=output/"records"/uid/role/(shift_id(dx,dy)+".json")
                r=read(path) if path.exists() else {"status":"PENDING"}
                rows.append({**r,"unit_id":uid,"role":role,"dx_rgb":dx,"dy_rgb":dy})
    result={"status":"COMPLETE_DIAGNOSTIC" if all(r["status"]=="SCORED" for r in rows) else "INCOMPLETE",
            "source_pairs":8,"planned_records":112,"records":rows,
            "claim_ceiling":"oracle_translation_mechanism_probe_no_blind_detector_or_formal_FPR_claim"}
    counts={}
    for role in ROLES:
        for dx,dy in SHIFTS:
            rs=[r for r in rows if r["role"]==role and (r["dx_rgb"],r["dy_rgb"])==(dx,dy)]
            ok=[r for r in rs if r["status"]=="SCORED"]
            counts[role+"/"+shift_id(dx,dy)]={"planned":8,"scored":len(ok),
                "pending":sum(r["status"]=="PENDING" for r in rs),
                "operational_error":sum(r["status"]=="OPERATIONAL_ERROR" for r in rs),
                "actual_production_above_reference_tau":sum(r["actual_production"]["above_reference_tau"] for r in ok)}
    result["counts"]=counts
    write(output/"summary.json",result)
    flat=[]
    for r in rows:
        f={k:r[k] for k in ("unit_id","role","dx_rgb","dy_rgb","status")}
        if r["status"]=="SCORED":
            f.update(r["metrics"])
            f["rgb_periodic_zero_identical"]=r["rgb_periodic_zero_identical"]
            f["actual_production_margin"]=r["actual_production"]["margin"]
            f["hf_full_actual"]=r["full_hf_actual"]["registered"]
            f["hf_full_base"]=r["full_hf_base"]["registered"]
            f["hf_common_actual"]=r["hf_diagnostic"]["actual_zero"]["registered"]
            f["hf_common_base"]=r["hf_diagnostic"]["base"]["registered"]
            f["hf_full_registered_loss"]=f["hf_full_base"]-f["hf_full_actual"]
            f["hf_full_gap_loss"]=r["full_hf_base"]["gap"]-r["full_hf_actual"]["gap"]
            for metric,changes in r["decomposition"].items():
                f.update({metric+"_"+k:v for k,v in changes.items()})
        flat.append(f)
    columns=list(dict.fromkeys(k for r in flat for k in r))
    tmp=output/"readout.csv.tmp"
    with tmp.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
    os.replace(tmp,output/"readout.csv")
    return result


def run(source,output,repo):
    import numpy as np
    import torch
    from PIL import Image
    from cegwm.shared.keys import normalize_detection_key,public_key_digest
    if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
        raise ValueError("SOURCE and OUTPUT must not overlap")
    if sha(support.__file__)!=SUPPORT_SHA:
        raise ValueError("Validated VAE-only support module identity differs")
    protocol,old=inspect_source(source)
    if subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()!=EXACT:
        raise ValueError("Historical source revision differs")
    if subprocess.check_output(["git","-C",str(repo),"diff","HEAD","--"],text=True):
        raise ValueError("Historical tracked source modified")
    if public_key_digest(normalize_detection_key(os.environ["CEG_WM_ROOT_KEY"]))!=protocol["key_digest"]:
        raise ValueError("Key differs from source run")
    identity={"schema":"crop_mechanism_v1","source":str(source.resolve()),"script_sha":sha(__file__),
              "support_sha":SUPPORT_SHA,"source_exact":EXACT,"shifts_rgb":[list(s) for s in SHIFTS],
              "source_hashes":{name:sha(source/name) for name in ("protocol.json","summary.json","environment.json","model_revisions.json")},
              "boundary":"encoded_black_residual_circular_vs_zero_extension",
              "phase":"Nyquist_row_column_removed_common_HF_subspace",
              "padding_latent":[32,64],"inner_latent":[8,56],"replay_atol":ATOL,"replay_rtol":RTOL}
    bind(output/"protocol.json",identity)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA VAE reader required; no generation fallback")
    environment={"python":platform.python_version(),"gpu":torch.cuda.get_device_name(0),"cuda":torch.version.cuda,
        "packages":{p:version(p) for p in ("torch","torchvision","diffusers","transformers","numpy","Pillow","huggingface_hub","safetensors")}}
    bind(output/"environment.json",environment)
    if summarize(output)["status"]=="COMPLETE_DIAGNOSTIC":
        print("Already complete; no scoring repeated.",flush=True)
        return
    reader=Reader(repo,source)
    black=reader.observe(Image.new("RGB",(512,512),(0,0,0))).float()
    if tuple(black.shape)!=(1,16,64,64):
        raise ValueError("Unexpected VAE observation shape")
    probe=HFProbe(reader,tuple(black.shape),black.device)
    refs={}
    # All 16 saved Oracle baselines must replay before the first mechanism probe.
    for uid in IDS:
        for role in ROLES:
            image=load_saved_image(source,old[uid,role,"oracle_h"]["image"])
            z=reader.observe(image)
            scored=reader.score(z)
            check=replay_comparison(scored,old[uid,role,"oracle_h"])
            full=probe.score(z,"full")
            hf_ok=np.allclose(list(full["scores"].values()),list(scored["branches"]["hf"].values()),atol=ATOL,rtol=RTOL)
            if not check["passed"] or not hf_ok:
                write(output/"replay_gate.json",{"status":"FAILED","unit_id":uid,"role":role,"check":check,"cached_HF_matches":bool(hf_ok)})
                raise RuntimeError("Baseline/HF scoring replay failed; no mechanism probes started")
            refs[uid,role]=(image,z,scored)
    write(output/"replay_gate.json",{"status":"PASSED","planned":16,"atol":ATOL,"rtol":RTOL})
    for uid in IDS:
        for role in ROLES:
            image,z0,base_production=refs[uid,role]
            for dx,dy in SHIFTS:
                path=output/"records"/uid/role/(shift_id(dx,dy)+".json")
                if path.exists() and read(path)["status"]=="SCORED": continue
                attempt=output/"attempts"/uid/role/shift_id(dx,dy)/(uuid.uuid4().hex+".json")
                write(attempt,{"status":"STARTED","time":time.time()})
                try:
                    zero=rgb_shift(image,dx,dy,False)
                    circular=rgb_shift(image,dx,dy,True)
                    same=bool(np.array_equal(np.asarray(zero),np.asarray(circular)))
                    actual=z0 if (dx,dy)==(0,0) else reader.observe(zero)
                    zcirc=actual if same else reader.observe(circular)
                    production=base_production if (dx,dy)==(0,0) else reader.score(actual)
                    fields,metrics=operator_fields(z0.float(),black,actual.float(),zcirc.float(),dx,dy)
                    diagnostic={name:probe.score(z) for name,z in fields.items()}
                    if (dx,dy)==(0,0):
                        for name,z in fields.items():
                            if relative(project(z),project(z0))>3e-5:
                                raise RuntimeError(f"Zero-shift control differs: {name}")
                    result={"status":"SCORED","rgb_shift_is_exact_integer":True,
                            "rgb_periodic_zero_identical":same,"actual_production":production,
                            "full_hf_actual":probe.score(actual,"full"),
                            "full_hf_base":probe.score(z0,"full"),"hf_diagnostic":diagnostic,
                            "metrics":metrics,"decomposition":decompose(diagnostic)}
                    write(attempt,result)
                    write(path,result)
                    changes=result["decomposition"]["registered"]
                    print(f"{uid[-4:]} {role} {shift_id(dx,dy)}: HF total={changes['total_loss']:.5f}, phase={changes['phase_loss']:.5f}, extra={changes['extra_vs_padded_phase']:.5f}",flush=True)
                except Exception as error:
                    failed={"status":"OPERATIONAL_ERROR",**error_data(error)}
                    write(attempt,failed);write(path,failed);summarize(output)
                    raise
            summarize(output)
    final=summarize(output)
    print(__import__("json").dumps({"status":final["status"],"counts":final["counts"]},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    for name in ("source","output","repo"): parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.source.resolve()) or args.source.resolve().is_relative_to(args.output.resolve()):
        raise ValueError("SOURCE and OUTPUT must not overlap")
    sys.path[:0]=[str(args.repo),str(args.repo/"src")]
    try: run(args.source,args.output,args.repo)
    except Exception as error:
        write(args.output/"last_failure.json",error_data(error))
        raise


if __name__=="__main__": main()
