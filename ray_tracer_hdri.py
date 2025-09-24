import triton
import triton.language as tl
import torch
import math
import numpy as np
try:
    import OpenEXR
    import Imath
except Exception:
    OpenEXR = None
    Imath = None

# Tune block size
BLOCK: int = 256
CONV_BLOCK: int = 1024

@triton.jit
def tonemap_kernel(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    v = tl.load(src_ptr + offs, mask=mask, other=0.0)
    v = tl.minimum(tl.maximum(v, 0.0), 1.0) * 255.0
    tl.store(dst_ptr + offs, v.to(tl.uint8), mask=mask)

@triton.jit
def raytrace_kernel(img_ptr,
                    eye_ptr, light_ptr,
                    sx_ptr, sy_ptr, sz_ptr, sr_ptr, refl_ptr, shine_ptr,
                    t_sin, t_cos, num_spheres,
                    rd_x_ptr, rd_y_ptr, rd_z_ptr,
                    in_hit_ptr,
                    refl_dir_x_ptr, refl_dir_y_ptr, refl_dir_z_ptr, refl_t_ptr,
                    hit_refl_ptr,
                    H: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    y = offs // W
    x = offs % W
    mask = (x < W) & (y < H)

    # Use pixel-center sampling and correct aspect ratio so images don't stretch
    # Avoid calling tl.float32(...) directly; use float literals so Triton promotes types
    x_f = x + 0.5
    y_f = y + 0.5
    u = (x_f / W) * 2.0 - 1.0
    v = (y_f / H) * 2.0 - 1.0
    # Scale horizontal coordinate by aspect ratio (width/height)
    u = u * (W / H)

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

    in_hit = closest_t < 1e9
    out_r = tl.zeros([BLOCK], dtype=tl.float32)
    out_g = tl.zeros([BLOCK], dtype=tl.float32)
    out_b = tl.zeros([BLOCK], dtype=tl.float32)
    # initialize reflection outputs so they exist even when there is no hit
    refl_dir_x = tl.zeros([BLOCK], dtype=tl.float32)
    refl_dir_y = tl.zeros([BLOCK], dtype=tl.float32)
    refl_dir_z = tl.zeros([BLOCK], dtype=tl.float32)
    refl_t = tl.zeros([BLOCK], dtype=tl.float32) + 1e9

    any_hit = tl.sum(in_hit.to(tl.int32), axis=0) > 0

    if any_hit:
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
        eps_spec = 1e-6
        log_term = tl.log2(spec_angle + eps_spec)
        specular = tl.exp2(hit_shine * log_term)
        base_color = ambient + diffuse + specular * 0.5

        out_r = tl.where(in_hit, base_color, out_r)
        out_g = tl.where(in_hit, base_color, out_g)
        out_b = tl.where(in_hit, base_color, out_b)

        dot_dn = rd_x * hit_nx + rd_y * hit_ny + rd_z * hit_nz
        refl_dir_x = rd_x - 2.0 * dot_dn * hit_nx
        refl_dir_y = rd_y - 2.0 * dot_dn * hit_ny
        refl_dir_z = rd_z - 2.0 * dot_dn * hit_nz
        refl_len = tl.sqrt(refl_dir_x * refl_dir_x + refl_dir_y * refl_dir_y + refl_dir_z * refl_dir_z)
        inv_refl = 1.0 / refl_len
        refl_dir_x *= inv_refl
        refl_dir_y *= inv_refl
        refl_dir_z *= inv_refl
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
            px2 = refl_origin_x + refl_dir_x * t_hit2
            py2 = refl_origin_y + refl_dir_y * t_hit2
            pz2 = refl_origin_z + refl_dir_z * t_hit2
            nx2 = (px2 - s_x) / s_r
            ny2 = (py2 - s_y) / s_r
            nz2 = (pz2 - s_z) / s_r
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

        out_r = tl.where(in_hit, out_r * (1.0 - hit_refl) + refl_val * hit_refl, out_r)
        out_g = tl.where(in_hit, out_g * (1.0 - hit_refl) + refl_val * hit_refl, out_g)
        out_b = tl.where(in_hit, out_b * (1.0 - hit_refl) + refl_val * hit_refl, out_b)

    offset = offs * 3
    # store RGB
    tl.store(img_ptr + offset + 0, out_r, mask=mask)
    tl.store(img_ptr + offset + 1, out_g, mask=mask)
    tl.store(img_ptr + offset + 2, out_b, mask=mask)

    # store primary ray direction and hit mask for host-side environment sampling
    tl.store(rd_x_ptr + offs, rd_x, mask=mask)
    tl.store(rd_y_ptr + offs, rd_y, mask=mask)
    tl.store(rd_z_ptr + offs, rd_z, mask=mask)
    # in_hit as float32 (0.0 or 1.0)
    tl.store(in_hit_ptr + offs, in_hit.to(tl.float32), mask=mask)

    # store reflection direction and reflection hit t (refl_t initialized to large if miss)
    tl.store(refl_dir_x_ptr + offs, refl_dir_x, mask=mask)
    tl.store(refl_dir_y_ptr + offs, refl_dir_y, mask=mask)
    tl.store(refl_dir_z_ptr + offs, refl_dir_z, mask=mask)
    tl.store(refl_t_ptr + offs, refl_t, mask=mask)

    # store per-pixel material reflectivity used in blending
    tl.store(hit_refl_ptr + offs, hit_refl, mask=mask)

# Camera & light
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

def render_frame(frame_index, width, height):
    H, W = height, width
    N_PIXELS = H * W
    N_FLOATS = N_PIXELS * 3
    eye_buf = eye.to(torch.float32)
    light_buf = light.to(torch.float32)

    grid = ((N_PIXELS + BLOCK - 1) // BLOCK,)
    conv_grid = ((N_FLOATS + CONV_BLOCK - 1) // CONV_BLOCK,)

    # Single buffers
    float_img = torch.empty(N_FLOATS, device="cuda", dtype=torch.float32)
    uint8_dev = torch.empty(N_FLOATS, device="cuda", dtype=torch.uint8)
    host_img = torch.empty((H, W, 3), dtype=torch.uint8, pin_memory=True)

    copy_stream = torch.cuda.Stream()
    copy_done = torch.cuda.Event(enable_timing=False)

    t_scalar = frame_index * 0.04
    t_sin = math.sin(t_scalar)
    t_cos = math.cos(t_scalar)

    # allocate per-pixel buffers for ray dirs, hits and reflections
    rd_x_dev = torch.empty(N_PIXELS, device="cuda", dtype=torch.float32)
    rd_y_dev = torch.empty(N_PIXELS, device="cuda", dtype=torch.float32)
    rd_z_dev = torch.empty(N_PIXELS, device="cuda", dtype=torch.float32)
    in_hit_dev = torch.empty(N_PIXELS, device="cuda", dtype=torch.float32)
    refl_dir_x_dev = torch.empty(N_PIXELS, device="cuda", dtype=torch.float32)
    refl_dir_y_dev = torch.empty(N_PIXELS, device="cuda", dtype=torch.float32)
    refl_dir_z_dev = torch.empty(N_PIXELS, device="cuda", dtype=torch.float32)
    refl_t_dev = torch.empty(N_PIXELS, device="cuda", dtype=torch.float32)
    hit_refl_dev = torch.empty(N_PIXELS, device="cuda", dtype=torch.float32)

    # Launch on default stream
    raytrace_kernel[grid](
        float_img,
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
        rd_x_dev,
        rd_y_dev,
        rd_z_dev,
        in_hit_dev,
        refl_dir_x_dev,
        refl_dir_y_dev,
        refl_dir_z_dev,
        refl_t_dev,
        hit_refl_dev,
        H,
        W,
        BLOCK
    )

    # load HDRI (cached)
    def _load_exr_to_cuda(path):
        if OpenEXR is None or Imath is None:
            # fallback to imageio if OpenEXR not available
            try:
                import imageio
            except Exception:
                raise RuntimeError("OpenEXR not available and imageio missing; cannot load EXR")
            img = imageio.v3.imread(path, format='EXR-FI')
            arr = np.asarray(img, dtype=np.float32)
            return torch.from_numpy(arr).to(device='cuda')
        f = OpenEXR.InputFile(path)
        dw = f.header()['dataWindow']
        W_exr = dw.max.x - dw.min.x + 1
        H_exr = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        chs = f.channels("RGB", pt)
        r = np.frombuffer(chs[0], dtype=np.float32).reshape(H_exr, W_exr)
        g = np.frombuffer(chs[1], dtype=np.float32).reshape(H_exr, W_exr)
        b = np.frombuffer(chs[2], dtype=np.float32).reshape(H_exr, W_exr)
        arr = np.stack([r, g, b], axis=2)
        return torch.from_numpy(arr).to(device='cuda')

    # cache HDRI tensor at module level to avoid reloading every frame
    global _HDRI_TENSOR
    if '_HDRI_TENSOR' not in globals():
        try:
            _HDRI_TENSOR = _load_exr_to_cuda('assets/hdri/golden_bay_1k.exr')
        except Exception:
            # fall back to a small constant environment if loading fails
            _HDRI_TENSOR = torch.ones((16, 32, 3), device='cuda', dtype=torch.float32) * 0.2

    hdr = _HDRI_TENSOR

    # helper: sample equirectangular HDR by direction vectors (all tensors on CUDA)
    def sample_equirect(hdr_tex, dir_x, dir_y, dir_z):
        Hh, Wh, _ = hdr_tex.shape
        # longitude (theta) around Y axis: atan2(x, z)
        lon = torch.atan2(dir_x, dir_z)
        lat = torch.asin(torch.clamp(dir_y, -1.0, 1.0))
        u = (lon / (2.0 * math.pi)) + 0.5
        v = 0.5 - (lat / math.pi)
        x = u * (Wh - 1)
        y = v * (Hh - 1)
        x0 = torch.floor(x).long() % Wh
        x1 = (x0 + 1) % Wh
        y0 = torch.clamp(torch.floor(y).long(), 0, Hh - 1)
        y1 = torch.clamp(y0 + 1, 0, Hh - 1)
        wx = (x - x0.float()).unsqueeze(1)
        wy = (y - y0.float()).unsqueeze(1)
        hdr_flat = hdr_tex.view(-1, 3)
        idx00 = (y0 * Wh + x0)
        idx10 = (y0 * Wh + x1)
        idx01 = (y1 * Wh + x0)
        idx11 = (y1 * Wh + x1)
        c00 = hdr_flat[idx00]
        c10 = hdr_flat[idx10]
        c01 = hdr_flat[idx01]
        c11 = hdr_flat[idx11]
        c0 = c00 * (1.0 - wx) + c10 * wx
        c1 = c01 * (1.0 - wx) + c11 * wx
        c = c0 * (1.0 - wy) + c1 * wy
        return c

    # reshape float image to (N_PIXELS, 3)
    img_dev = float_img.view(N_PIXELS, 3)

    # Primary misses: where in_hit == 0
    primary_miss = (in_hit_dev == 0.0)
    if primary_miss.any():
        env = sample_equirect(hdr, rd_x_dev, rd_y_dev, rd_z_dev)
        img_dev[primary_miss] = env[primary_miss]

    # Reflection misses: pixels that had a hit but reflection missed (large refl_t) and have reflectivity
    refl_miss = (in_hit_dev > 0.0) & (refl_t_dev > 1e8) & (hit_refl_dev > 0.0)
    if refl_miss.any():
        env_refl = sample_equirect(hdr, refl_dir_x_dev, refl_dir_y_dev, refl_dir_z_dev)
        # add env contribution scaled by material reflectivity
        add = env_refl * hit_refl_dev.unsqueeze(1)
        img_dev[refl_miss] = img_dev[refl_miss] + add[refl_miss]

    # now tonemap
    tonemap_kernel[conv_grid](
        float_img,
        uint8_dev,
        N_FLOATS,
        CONV_BLOCK
    )

    # Async copy to host pinned memory
    with torch.cuda.stream(copy_stream):
        host_img.view(-1).copy_(uint8_dev, non_blocking=True)
        copy_done.record(copy_stream)

    # Wait for copy to complete then write frame
    copy_done.synchronize()
    return host_img