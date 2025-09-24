import triton
import triton.language as tl
import torch
import math
import OpenEXR
import Imath
import numpy as np

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
def atan(x):
    pi_half = 1.5707963267948966
    abs_x = tl.abs(x)
    # For |x| <= 1, use Taylor series
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    x7 = x5 * x2
    res_small = x - x3 / 3 + x5 / 5 - x7 / 7
    # For |x| > 1, use atan(x) = pi/2 - atan(1/x) for x > 0, -pi/2 - atan(1/x) for x < 0
    inv_x = 1 / x
    inv_x2 = inv_x * inv_x
    inv_x3 = inv_x2 * inv_x
    inv_x5 = inv_x3 * inv_x2
    inv_x7 = inv_x5 * inv_x2
    atan_inv = inv_x - inv_x3 / 3 + inv_x5 / 5 - inv_x7 / 7
    res_large = tl.where(x > 0, pi_half - atan_inv, -pi_half - atan_inv)
    return tl.where(abs_x <= 1, res_small, res_large)

@triton.jit
def atan2(y, x):
    pi = 3.141592653589793
    res = atan(y / x)
    res = tl.where(x < 0, res + tl.where(y >= 0, pi, -pi), res)
    return res

@triton.jit
def dir_to_uv(rd_x, rd_y, rd_z):
    theta = atan2( tl.sqrt(1 - rd_z*rd_z), rd_z )
    phi = atan2(rd_y, rd_x)
    u = (phi + 3.141592653589793) / (2 * 3.141592653589793)
    v = theta / 3.141592653589793
    return u, v

@triton.jit
def sample_hdri(hdri_ptr, u, v, hdri_w, hdri_h):
    x = u * (hdri_w - 1)
    y = v * (hdri_h - 1)
    x0 = tl.floor(x)
    y0 = tl.floor(y)
    x1 = tl.minimum(x0 + 1, hdri_w - 1)
    y1 = tl.minimum(y0 + 1, hdri_h - 1)
    fx = x - x0
    fy = y - y0
    idx00 = ((y0 * hdri_w + x0) * 3).to(tl.int32)
    idx01 = ((y0 * hdri_w + x1) * 3).to(tl.int32)
    idx10 = ((y1 * hdri_w + x0) * 3).to(tl.int32)
    idx11 = ((y1 * hdri_w + x1) * 3).to(tl.int32)
    r00 = tl.load(hdri_ptr + idx00 + 0)
    g00 = tl.load(hdri_ptr + idx00 + 1)
    b00 = tl.load(hdri_ptr + idx00 + 2)
    r01 = tl.load(hdri_ptr + idx01 + 0)
    g01 = tl.load(hdri_ptr + idx01 + 1)
    b01 = tl.load(hdri_ptr + idx01 + 2)
    r10 = tl.load(hdri_ptr + idx10 + 0)
    g10 = tl.load(hdri_ptr + idx10 + 1)
    b10 = tl.load(hdri_ptr + idx10 + 2)
    r11 = tl.load(hdri_ptr + idx11 + 0)
    g11 = tl.load(hdri_ptr + idx11 + 1)
    b11 = tl.load(hdri_ptr + idx11 + 2)
    r = (1-fx)*(1-fy)*r00 + fx*(1-fy)*r01 + (1-fx)*fy*r10 + fx*fy*r11
    g = (1-fx)*(1-fy)*g00 + fx*(1-fy)*g01 + (1-fx)*fy*g10 + fx*fy*g11
    b = (1-fx)*(1-fy)*b00 + fx*(1-fy)*b01 + (1-fx)*fy*b10 + fx*fy*b11
    return r, g, b

@triton.jit
def raytrace_kernel(img_ptr,
                    eye_ptr, light_ptr,
                    sx_ptr, sy_ptr, sz_ptr, sr_ptr, refl_ptr, shine_ptr,
                    t_sin, t_cos, num_spheres,
                    hdri_ptr, hdri_w, hdri_h,
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

    bg_u, bg_v = dir_to_uv(rd_x, rd_y, rd_z)
    bg_r, bg_g, bg_b = sample_hdri(hdri_ptr, bg_u, bg_v, hdri_w, hdri_h)

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
    out_r = tl.where(~in_hit, bg_r, tl.zeros([BLOCK], dtype=tl.float32))
    out_g = tl.where(~in_hit, bg_g, tl.zeros([BLOCK], dtype=tl.float32))
    out_b = tl.where(~in_hit, bg_b, tl.zeros([BLOCK], dtype=tl.float32))

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
        refl_r = tl.zeros([BLOCK], dtype=tl.float32)
        refl_g = tl.zeros([BLOCK], dtype=tl.float32)
        refl_b = tl.zeros([BLOCK], dtype=tl.float32)
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
            refl_r = tl.where(valid2, diff2, refl_r)
            refl_g = tl.where(valid2, diff2, refl_g)
            refl_b = tl.where(valid2, diff2, refl_b)
        refl_u, refl_v = dir_to_uv(refl_dir_x, refl_dir_y, refl_dir_z)
        no_refl_hit = refl_t >= 1e9
        refl_bg_r, refl_bg_g, refl_bg_b = sample_hdri(hdri_ptr, refl_u, refl_v, hdri_w, hdri_h)
        refl_r = tl.where(no_refl_hit, refl_bg_r, refl_r)
        refl_g = tl.where(no_refl_hit, refl_bg_g, refl_g)
        refl_b = tl.where(no_refl_hit, refl_bg_b, refl_b)
        out_r = tl.where(in_hit, out_r * (1.0 - hit_refl) + refl_r * hit_refl, out_r)
        out_g = tl.where(in_hit, out_g * (1.0 - hit_refl) + refl_g * hit_refl, out_g)
        out_b = tl.where(in_hit, out_b * (1.0 - hit_refl) + refl_b * hit_refl, out_b)

    offset = offs * 3
    tl.store(img_ptr + offset + 0, out_r, mask=mask)
    tl.store(img_ptr + offset + 1, out_g, mask=mask)
    tl.store(img_ptr + offset + 2, out_b, mask=mask)

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

def load_hdri(path):
    file = OpenEXR.InputFile(path)
    dw = file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    rgb = [np.frombuffer(file.channel(c, FLOAT), dtype=np.float32).reshape(height, width) for c in "RGB"]
    hdri = np.stack(rgb, axis=-1)
    
    # Rotate HDRI 90 degrees around X axis to orient landscape properly
    # Create coordinate grids
    j_grid, i_grid = np.meshgrid(np.arange(width), np.arange(height))
    u = (j_grid + 0.5) / width
    v = (i_grid + 0.5) / height
    phi = (u - 0.5) * 2 * np.pi
    theta = v * np.pi
    
    # Direction vectors
    sin_theta = np.sin(theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = np.cos(theta)
    
    # Apply 90 deg rotation around X axis: (x, y, z) -> (x, -z, y)
    rot_x = x
    rot_y = z
    rot_z = -y
    
    # Compute new UV coordinates
    rot_theta = np.arccos(np.clip(rot_z, -1, 1))
    rot_phi = np.arctan2(rot_y, rot_x)
    rot_u = (rot_phi + np.pi) / (2 * np.pi)
    rot_v = rot_theta / np.pi
    
    # Bilinear sampling
    rot_u = np.clip(rot_u, 0, 1)
    rot_v = np.clip(rot_v, 0, 1)
    x_f = rot_u * (width - 1)
    y_f = rot_v * (height - 1)
    x0 = np.floor(x_f).astype(int)
    y0 = np.floor(y_f).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = x_f - x0
    fy = y_f - y0
    
    rotated_hdri = np.zeros_like(hdri)
    for c in range(3):
        v00 = hdri[y0, x0, c]
        v01 = hdri[y0, x1, c]
        v10 = hdri[y1, x0, c]
        v11 = hdri[y1, x1, c]
        rotated_hdri[i_grid, j_grid, c] = (1-fx)*(1-fy)*v00 + fx*(1-fy)*v01 + (1-fx)*fy*v10 + fx*fy*v11
    
    return rotated_hdri

hdri = load_hdri('assets/hdri/golden_bay_1k.exr')
hdri_tensor = torch.from_numpy(hdri).cuda()
hdri_height, hdri_width, _ = hdri.shape

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
        hdri_tensor,
        hdri_width,
        hdri_height,
        H,
        W,
        BLOCK
    )
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