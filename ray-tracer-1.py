import triton
import triton.language as tl
import torch
from PIL import Image
import time
import math
import cv2

# Image size
H, W = 1024, 1024

eye = torch.tensor([0.0, 0.0, -3.0], device="cuda")
light = torch.tensor([5.0, 5.0, -5.0], device="cuda")

# Spheres: (x,y,z,r,reflectivity,shininess)
base_spheres = torch.tensor([
    [0.0, 0.0, 0.0, 0.5, 0.3, 64.0],   # animated sin
    [1.0, 0.0, 0.0, 0.5, 0.6, 128.0],  # animated cos
    [-0.75, -0.3, 0.2, 0.3, 0.1, 32.0],# static small
    [0.6, 0.5, -0.2, 0.4, 0.4, 96.0],  # static
], device="cuda", dtype=torch.float32)
NUM_SPHERES = base_spheres.shape[0]

sphere_x = base_spheres[:, 0].contiguous()
sphere_y = base_spheres[:, 1].contiguous()
sphere_z = base_spheres[:, 2].contiguous()
sphere_r = base_spheres[:, 3].contiguous()
sphere_reflect = base_spheres[:, 4].contiguous()
sphere_shine = base_spheres[:, 5].contiguous()

# Tune block size (number of pixels processed per program). 256 is a common starting point.
BLOCK: int = 256
CONV_BLOCK: int = 1024  # separate block size for tonemap kernel

@triton.jit
def tonemap_kernel(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    v = tl.load(src_ptr + offs, mask=mask, other=0.0)
    # clamp 0..1, scale, convert
    v = tl.minimum(tl.maximum(v, 0.0), 1.0) * 255.0
    tl.store(dst_ptr + offs, v.to(tl.uint8), mask=mask)

@triton.jit
def raytrace_kernel(img_ptr,
                    eye_ptr, light_ptr,
                    sx_ptr, sy_ptr, sz_ptr, sr_ptr, refl_ptr, shine_ptr,
                    t_sin, t_cos, num_spheres,
                    H: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    y = offs // W
    x = offs % W
    mask = (x < W) & (y < H)

    u = (x / W) * 2 - 1
    v = (y / H) * 2 - 1

    rd_x = u
    rd_y = v
    rd_z = 1.0
    inv_len = 1.0 / tl.sqrt(rd_x * rd_x + rd_y * rd_y + rd_z * rd_z)
    rd_x *= inv_len
    rd_y *= inv_len
    rd_z *= inv_len

    eye_x = tl.load(eye_ptr + 0)
    eye_y = tl.load(eye_ptr + 1)
    eye_z = tl.load(eye_ptr + 2)
    light_x = tl.load(light_ptr + 0)
    light_y = tl.load(light_ptr + 1)
    light_z = tl.load(light_ptr + 2)

    closest_t = tl.zeros([BLOCK], dtype=tl.float32) + 1e9
    hit_px = tl.zeros([BLOCK], dtype=tl.float32)
    hit_py = tl.zeros([BLOCK], dtype=tl.float32)
    hit_pz = tl.zeros([BLOCK], dtype=tl.float32)
    hit_nx = tl.zeros([BLOCK], dtype=tl.float32)
    hit_ny = tl.zeros([BLOCK], dtype=tl.float32)
    hit_nz = tl.zeros([BLOCK], dtype=tl.float32)
    hit_refl = tl.zeros([BLOCK], dtype=tl.float32)
    hit_shine = tl.zeros([BLOCK], dtype=tl.float32)

    # Loop spheres
    for i in range(0, num_spheres):
        s_x = tl.load(sx_ptr + i)
        s_y = tl.load(sy_ptr + i)
        s_z = tl.load(sz_ptr + i)
        s_r = tl.load(sr_ptr + i)
        s_refl = tl.load(refl_ptr + i)
        s_shine = tl.load(shine_ptr + i)
        if i == 0:
            s_x = t_sin * 0.5
        if i == 1:
            s_x = 1.0 + t_cos * 0.5
        oc_x = eye_x - s_x
        oc_y = eye_y - s_y
        oc_z = eye_z - s_z
        b = 2.0 * (oc_x * rd_x + oc_y * rd_y + oc_z * rd_z)
        c = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z - s_r * s_r
        disc = b * b - 4.0 * c
        hit_mask = disc > 0
        sqrt_disc = tl.where(hit_mask, tl.sqrt(disc), 0.0)
        t_hit = (-b - sqrt_disc) * 0.5
        valid = hit_mask & (t_hit > 0) & (t_hit < closest_t)
        # Update closest
        closest_t = tl.where(valid, t_hit, closest_t)
        px = eye_x + rd_x * t_hit
        py = eye_y + rd_y * t_hit
        pz = eye_z + rd_z * t_hit
        nx = (px - s_x) / s_r
        ny = (py - s_y) / s_r
        nz = (pz - s_z) / s_r
        hit_px = tl.where(valid, px, hit_px)
        hit_py = tl.where(valid, py, hit_py)
        hit_pz = tl.where(valid, pz, hit_pz)
        hit_nx = tl.where(valid, nx, hit_nx)
        hit_ny = tl.where(valid, ny, hit_ny)
        hit_nz = tl.where(valid, nz, hit_nz)
        hit_refl = tl.where(valid, s_refl, hit_refl)
        hit_shine = tl.where(valid, s_shine, hit_shine)

    # Lighting & shadows
    in_hit = closest_t < 1e9
    # Initialize output colors
    out_r = tl.zeros([BLOCK], dtype=tl.float32)
    out_g = tl.zeros([BLOCK], dtype=tl.float32)
    out_b = tl.zeros([BLOCK], dtype=tl.float32)

    # Check if any lane hit by summing boolean mask (True->1, False->0)
    any_hit = tl.sum(in_hit.to(tl.int32), axis=0) > 0

    if any_hit:
        # Diffuse + shadow
        l_x = light_x - hit_px
        l_y = light_y - hit_py
        l_z = light_z - hit_pz
        l_len = tl.sqrt(l_x * l_x + l_y * l_y + l_z * l_z)
        inv_l = 1.0 / l_len
        l_x *= inv_l
        l_y *= inv_l
        l_z *= inv_l
        diff = hit_nx * l_x + hit_ny * l_y + hit_nz * l_z
        diff = tl.maximum(diff, 0.0)
        eps = 1e-3
        shadow_origin_x = hit_px + hit_nx * eps
        shadow_origin_y = hit_py + hit_ny * eps
        shadow_origin_z = hit_pz + hit_nz * eps
        shadow = tl.zeros([BLOCK], dtype=tl.float32) + 1.0
        for i in range(0, num_spheres):
            active = (shadow > 0) & in_hit
            # If no active lanes remain, still iterate but do nothing (masked ops)
            s_x = tl.load(sx_ptr + i)
            s_y = tl.load(sy_ptr + i)
            s_z = tl.load(sz_ptr + i)
            s_r = tl.load(sr_ptr + i)
            if i == 0:
                s_x = t_sin * 0.5
            if i == 1:
                s_x = 1.0 + t_cos * 0.5
            oc_x = shadow_origin_x - s_x
            oc_y = shadow_origin_y - s_y
            oc_z = shadow_origin_z - s_z
            b = 2.0 * (oc_x * l_x + oc_y * l_y + oc_z * l_z)
            c = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z - s_r * s_r
            disc = b * b - 4.0 * c
            m = (disc > 0) & active
            sqrt_disc = tl.where(m, tl.sqrt(disc), 0.0)
            t_sh = (-b - sqrt_disc) * 0.5
            blocker = m & (t_sh > 0.0) & (t_sh < l_len)
            shadow = tl.where(blocker, 0.0, shadow)
        ambient = 0.05
        diffuse = diff * shadow
        # Specular (Blinn-Phong)
        view_x = eye_x - hit_px
        view_y = eye_y - hit_py
        view_z = eye_z - hit_pz
        v_len = tl.sqrt(view_x * view_x + view_y * view_y + view_z * view_z)
        inv_v = 1.0 / v_len
        view_x *= inv_v
        view_y *= inv_v
        view_z *= inv_v
        half_x = view_x + l_x
        half_y = view_y + l_y
        half_z = view_z + l_z
        h_len = tl.sqrt(half_x * half_x + half_y * half_y + half_z * half_z)
        inv_h = 1.0 / h_len
        half_x *= inv_h
        half_y *= inv_h
        half_z *= inv_h
        spec_angle = hit_nx * half_x + hit_ny * half_y + hit_nz * half_z
        spec_angle = tl.maximum(spec_angle, 0.0)
        # Specular using exp2(hit_shine * log2(spec_angle + eps)) to avoid unsupported power op
        eps_spec = 1e-6
        log_term = tl.log2(spec_angle + eps_spec)
        specular = tl.exp2(hit_shine * log_term)
        base_color = ambient + diffuse + specular * 0.5
        out_r = tl.where(in_hit, base_color, out_r)
        out_g = tl.where(in_hit, base_color, out_g)
        out_b = tl.where(in_hit, base_color, out_b)

        # Single-bounce reflections
        # Compute reflection ray R = D - 2 (D·N) N
        dot_dn = rd_x * hit_nx + rd_y * hit_ny + rd_z * hit_nz
        refl_dir_x = rd_x - 2.0 * dot_dn * hit_nx
        refl_dir_y = rd_y - 2.0 * dot_dn * hit_ny
        refl_dir_z = rd_z - 2.0 * dot_dn * hit_nz
        # Normalize reflection
        refl_len = tl.sqrt(refl_dir_x * refl_dir_x + refl_dir_y * refl_dir_y + refl_dir_z * refl_dir_z)
        inv_refl = 1.0 / refl_len
        refl_dir_x *= inv_refl
        refl_dir_y *= inv_refl
        refl_dir_z *= inv_refl
        # Trace reflection against spheres (no further reflections, no shadows to save cost)
        refl_t = tl.zeros([BLOCK], dtype=tl.float32) + 1e9
        refl_val = tl.zeros([BLOCK], dtype=tl.float32)
        refl_origin_x = hit_px + hit_nx * eps
        refl_origin_y = hit_py + hit_ny * eps
        refl_origin_z = hit_pz + hit_nz * eps
        for i in range(0, num_spheres):
            s_x = tl.load(sx_ptr + i)
            s_y = tl.load(sy_ptr + i)
            s_z = tl.load(sz_ptr + i)
            s_r = tl.load(sr_ptr + i)
            if i == 0:
                s_x = t_sin * 0.5
            if i == 1:
                s_x = 1.0 + t_cos * 0.5
            oc_x = refl_origin_x - s_x
            oc_y = refl_origin_y - s_y
            oc_z = refl_origin_z - s_z
            b = 2.0 * (oc_x * refl_dir_x + oc_y * refl_dir_y + oc_z * refl_dir_z)
            c = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z - s_r * s_r
            disc = b * b - 4.0 * c
            hitm = disc > 0
            sqrt_disc = tl.where(hitm, tl.sqrt(disc), 0.0)
            t_hit2 = (-b - sqrt_disc) * 0.5
            valid2 = hitm & (t_hit2 > 0) & (t_hit2 < refl_t)
            # For reflection brightness we just reuse diffuse-style term with light
            px2 = refl_origin_x + refl_dir_x * t_hit2
            py2 = refl_origin_y + refl_dir_y * t_hit2
            pz2 = refl_origin_z + refl_dir_z * t_hit2
            nx2 = (px2 - s_x) / s_r
            ny2 = (py2 - s_y) / s_r
            nz2 = (pz2 - s_z) / s_r
            # Light vector
            l2x = light_x - px2
            l2y = light_y - py2
            l2z = light_z - pz2
            l2len = tl.sqrt(l2x * l2x + l2y * l2y + l2z * l2z)
            inv_l2 = 1.0 / l2len
            l2x *= inv_l2
            l2y *= inv_l2
            l2z *= inv_l2
            diff2 = nx2 * l2x + ny2 * l2y + nz2 * l2z
            diff2 = tl.maximum(diff2, 0.0)
            refl_t = tl.where(valid2, t_hit2, refl_t)
            refl_val = tl.where(valid2, diff2, refl_val)
        # Mix reflection
        out_r = tl.where(in_hit, out_r * (1.0 - hit_refl) + refl_val * hit_refl, out_r)
        out_g = tl.where(in_hit, out_g * (1.0 - hit_refl) + refl_val * hit_refl, out_g)
        out_b = tl.where(in_hit, out_b * (1.0 - hit_refl) + refl_val * hit_refl, out_b)

    # Store
    offset = offs * 3
    tl.store(img_ptr + offset + 0, out_r, mask=mask)
    tl.store(img_ptr + offset + 1, out_g, mask=mask)
    tl.store(img_ptr + offset + 2, out_b, mask=mask)

N_PIXELS = H * W
N_FLOATS = N_PIXELS * 3
eye_buf = eye.to(torch.float32)
light_buf = light.to(torch.float32)

grid = ((N_PIXELS + BLOCK - 1) // BLOCK,)
conv_grid = ((N_FLOATS + CONV_BLOCK - 1) // CONV_BLOCK,)

# Double buffers (float output + uint8 tonemapped)
float_imgs = [torch.empty(N_FLOATS, device="cuda", dtype=torch.float32),
              torch.empty(N_FLOATS, device="cuda", dtype=torch.float32)]
uint8_dev = [torch.empty(N_FLOATS, device="cuda", dtype=torch.uint8),
             torch.empty(N_FLOATS, device="cuda", dtype=torch.uint8)]
host_imgs = [torch.empty((H, W, 3), dtype=torch.uint8, pin_memory=True),
             torch.empty((H, W, 3), dtype=torch.uint8, pin_memory=True)]

copy_stream = torch.cuda.Stream()
copy_done = [torch.cuda.Event(enable_timing=False), torch.cuda.Event(enable_timing=False)]

cv2.namedWindow("ray", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ray", W, H)

save_first_frame = True
frame = 0
t0 = time.time()
fps_avg_window = 60
render_times = []
start_ev = torch.cuda.Event(enable_timing=True)
end_ev = torch.cuda.Event(enable_timing=True)

try:
    while True:
        buf = frame & 1
        prev = buf ^ 1
        t_scalar = frame * 0.04
        t_sin = math.sin(t_scalar)
        t_cos = math.cos(t_scalar)

        # Launch ray tracing + tonemap on default stream
        start_ev.record()
        raytrace_kernel[grid](
            float_imgs[buf],
            eye_buf,
            light_buf,
            sphere_x,
            sphere_y,
            sphere_z,
            sphere_r,
            sphere_reflect,
            sphere_shine,
            t_sin,
            t_cos,
            NUM_SPHERES,
            H,
            W,
            BLOCK
        )
        tonemap_kernel[conv_grid](
            float_imgs[buf],
            uint8_dev[buf],
            N_FLOATS,
            CONV_BLOCK
        )
        end_ev.record()

        # Async copy (convert layout) on separate stream
        with torch.cuda.stream(copy_stream):
            host_imgs[buf].view(-1).copy_(uint8_dev[buf], non_blocking=True)
            copy_done[buf].record(copy_stream)

        if frame > 0:
            # Ensure previous frame copy finished before display
            copy_done[prev].synchronize()
            frame_img_bgr = host_imgs[prev].numpy()[..., ::-1]
            cv2.imshow("ray", frame_img_bgr)
            if save_first_frame and frame == 1:
                Image.fromarray(host_imgs[prev].numpy()).save("out.png")
                print("Saved first frame to out.png")
                save_first_frame = False

        # Gather timing every 30 frames (avoids per-frame sync)
        if frame % 30 == 29:
            end_ev.synchronize()
            render_ms = start_ev.elapsed_time(end_ev)
            render_times.append(render_ms)
            if len(render_times) > fps_avg_window:
                render_times.pop(0)
            avg_ms = sum(render_times)/len(render_times)
            print(f"[Frame {frame}] kernel+tonemap avg {avg_ms:.3f} ms (~{1000.0/avg_ms:.1f} FPS) display ~{frame/(time.time()-t0):.1f} FPS")

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frame += 1
except KeyboardInterrupt:
    pass
finally:
    total = time.time() - t0
    if frame > 0:
        print(f"Stopped. {frame} frames in {total:.2f}s => {frame/total:.2f} displayed FPS (no workload reduction).")
    cv2.destroyAllWindows()
