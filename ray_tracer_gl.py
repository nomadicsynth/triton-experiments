import math, time, ctypes
import torch, triton, triton.language as tl
import glfw
from OpenGL import GL as gl

H, W = 1024, 1024
BLOCK = 256
CONV_BLOCK = 1024

# Scene setup matches ray-tracer-1.py
eye = torch.tensor([0.0, 0.0, -3.0], device='cuda')
light = torch.tensor([5.0, 5.0, -5.0], device='cuda')
base_spheres = torch.tensor([
    [0.0, 0.0, 0.0, 0.5, 0.3, 64.0],
    [1.0, 0.0, 0.0, 0.5, 0.6, 128.0],
    [-0.75, -0.3, 0.2, 0.3, 0.1, 32.0],
    [0.6, 0.5, -0.2, 0.4, 0.4, 96.0],
], device='cuda', dtype=torch.float32)
NUM_SPHERES = base_spheres.shape[0]

sphere_x = base_spheres[:,0].contiguous()
sphere_y = base_spheres[:,1].contiguous()
sphere_z = base_spheres[:,2].contiguous()
sphere_r = base_spheres[:,3].contiguous()
sphere_reflect = base_spheres[:,4].contiguous()
sphere_shine = base_spheres[:,5].contiguous()

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
    rd_x = u; rd_y = v; rd_z = 1.0
    inv_len = 1.0 / tl.sqrt(rd_x*rd_x + rd_y*rd_y + rd_z*rd_z)
    rd_x *= inv_len; rd_y *= inv_len; rd_z *= inv_len
    eye_x = tl.load(eye_ptr + 0); eye_y = tl.load(eye_ptr + 1); eye_z = tl.load(eye_ptr + 2)
    light_x = tl.load(light_ptr + 0); light_y = tl.load(light_ptr + 1); light_z = tl.load(light_ptr + 2)
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
        s_x = tl.load(sx_ptr + i); s_y = tl.load(sy_ptr + i); s_z = tl.load(sz_ptr + i)
        s_r = tl.load(sr_ptr + i); s_refl = tl.load(refl_ptr + i); s_shine = tl.load(shine_ptr + i)
        if i == 0: s_x = t_sin * 0.5
        if i == 1: s_x = 1.0 + t_cos * 0.5
        oc_x = eye_x - s_x; oc_y = eye_y - s_y; oc_z = eye_z - s_z
        b = 2.0*(oc_x*rd_x + oc_y*rd_y + oc_z*rd_z)
        c = oc_x*oc_x + oc_y*oc_y + oc_z*oc_z - s_r*s_r
        disc = b*b - 4.0*c
        hit_mask = disc > 0
        sqrt_disc = tl.where(hit_mask, tl.sqrt(disc), 0.0)
        t_hit = (-b - sqrt_disc) * 0.5
        valid = hit_mask & (t_hit > 0) & (t_hit < closest_t)
        closest_t = tl.where(valid, t_hit, closest_t)
        px = eye_x + rd_x * t_hit; py = eye_y + rd_y * t_hit; pz = eye_z + rd_z * t_hit
        nx = (px - s_x) / s_r; ny = (py - s_y) / s_r; nz = (pz - s_z) / s_r
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
    any_hit = tl.sum(in_hit.to(tl.int32), axis=0) > 0
    if any_hit:
        l_x = light_x - hit_px; l_y = light_y - hit_py; l_z = light_z - hit_pz
        l_len = tl.sqrt(l_x*l_x + l_y*l_y + l_z*l_z); inv_l = 1.0 / l_len
        l_x*=inv_l; l_y*=inv_l; l_z*=inv_l
        diff = hit_nx*l_x + hit_ny*l_y + hit_nz*l_z; diff = tl.maximum(diff, 0.0)
        eps = 1e-3
        shadow_origin_x = hit_px + hit_nx*eps
        shadow_origin_y = hit_py + hit_ny*eps
        shadow_origin_z = hit_pz + hit_nz*eps
        shadow = tl.zeros([BLOCK], dtype=tl.float32) + 1.0
        for i in range(0, num_spheres):
            active = (shadow > 0) & in_hit
            s_x = tl.load(sx_ptr + i); s_y = tl.load(sy_ptr + i); s_z = tl.load(sz_ptr + i); s_r = tl.load(sr_ptr + i)
            if i == 0: s_x = t_sin * 0.5
            if i == 1: s_x = 1.0 + t_cos * 0.5
            oc_x = shadow_origin_x - s_x; oc_y = shadow_origin_y - s_y; oc_z = shadow_origin_z - s_z
            b = 2.0*(oc_x*l_x + oc_y*l_y + oc_z*l_z)
            c = oc_x*oc_x + oc_y*oc_y + oc_z*oc_z - s_r*s_r
            disc = b*b - 4.0*c
            m = (disc > 0) & active
            sqrt_disc = tl.where(m, tl.sqrt(disc), 0.0)
            t_sh = (-b - sqrt_disc) * 0.5
            blocker = m & (t_sh > 0.0) & (t_sh < l_len)
            shadow = tl.where(blocker, 0.0, shadow)
        ambient = 0.05
        # specular
        view_x = eye_x - hit_px; view_y = eye_y - hit_py; view_z = eye_z - hit_pz
        v_len = tl.sqrt(view_x*view_x + view_y*view_y + view_z*view_z); inv_v = 1.0 / v_len
        view_x*=inv_v; view_y*=inv_v; view_z*=inv_v
        half_x = view_x + l_x; half_y = view_y + l_y; half_z = view_z + l_z
        h_len = tl.sqrt(half_x*half_x + half_y*half_y + half_z*half_z); inv_h = 1.0 / h_len
        half_x*=inv_h; half_y*=inv_h; half_z*=inv_h
        spec_angle = hit_nx*half_x + hit_ny*half_y + hit_nz*half_z
        spec_angle = tl.maximum(spec_angle, 0.0)
        eps_spec = 1e-6
        log_term = tl.log2(spec_angle + eps_spec)
        specular = tl.exp2(hit_shine * log_term)
        base_color = ambient + diff*shadow + specular*0.5
        out_r = tl.where(in_hit, base_color, out_r)
        out_g = tl.where(in_hit, base_color, out_g)
        out_b = tl.where(in_hit, base_color, out_b)

        # Reflections (single bounce) ported from original
        dot_dn = rd_x * hit_nx + rd_y * hit_ny + rd_z * hit_nz
        refl_dir_x = rd_x - 2.0 * dot_dn * hit_nx
        refl_dir_y = rd_y - 2.0 * dot_dn * hit_ny
        refl_dir_z = rd_z - 2.0 * dot_dn * hit_nz
        refl_len = tl.sqrt(refl_dir_x*refl_dir_x + refl_dir_y*refl_dir_y + refl_dir_z*refl_dir_z)
        inv_refl = 1.0 / refl_len
        refl_dir_x*=inv_refl; refl_dir_y*=inv_refl; refl_dir_z*=inv_refl
        refl_t = tl.zeros([BLOCK], dtype=tl.float32) + 1e9
        refl_val = tl.zeros([BLOCK], dtype=tl.float32)
        refl_origin_x = hit_px + hit_nx * 1e-3
        refl_origin_y = hit_py + hit_ny * 1e-3
        refl_origin_z = hit_pz + hit_nz * 1e-3
        for i in range(0, num_spheres):
            s_x = tl.load(sx_ptr + i); s_y = tl.load(sy_ptr + i); s_z = tl.load(sz_ptr + i); s_r = tl.load(sr_ptr + i)
            if i == 0: s_x = t_sin * 0.5
            if i == 1: s_x = 1.0 + t_cos * 0.5
            oc_x = refl_origin_x - s_x; oc_y = refl_origin_y - s_y; oc_z = refl_origin_z - s_z
            b2 = 2.0*(oc_x*refl_dir_x + oc_y*refl_dir_y + oc_z*refl_dir_z)
            c2 = oc_x*oc_x + oc_y*oc_y + oc_z*oc_z - s_r*s_r
            disc2 = b2*b2 - 4.0*c2
            hm = disc2 > 0
            sqrt_disc2 = tl.where(hm, tl.sqrt(disc2), 0.0)
            t_hit2 = (-b2 - sqrt_disc2) * 0.5
            valid2 = hm & (t_hit2 > 0) & (t_hit2 < refl_t)
            px2 = refl_origin_x + refl_dir_x * t_hit2
            py2 = refl_origin_y + refl_dir_y * t_hit2
            pz2 = refl_origin_z + refl_dir_z * t_hit2
            nx2 = (px2 - s_x) / s_r
            ny2 = (py2 - s_y) / s_r
            nz2 = (pz2 - s_z) / s_r
            l2x = light_x - px2; l2y = light_y - py2; l2z = light_z - pz2
            l2len = tl.sqrt(l2x*l2x + l2y*l2y + l2z*l2z); inv_l2 = 1.0 / l2len
            l2x*=inv_l2; l2y*=inv_l2; l2z*=inv_l2
            diff2 = nx2*l2x + ny2*l2y + nz2*l2z; diff2 = tl.maximum(diff2, 0.0)
            refl_t = tl.where(valid2, t_hit2, refl_t)
            refl_val = tl.where(valid2, diff2, refl_val)
        out_r = tl.where(in_hit, out_r*(1.0-hit_refl) + refl_val*hit_refl, out_r)
        out_g = tl.where(in_hit, out_g*(1.0-hit_refl) + refl_val*hit_refl, out_g)
        out_b = tl.where(in_hit, out_b*(1.0-hit_refl) + refl_val*hit_refl, out_b)
    offset = offs * 3
    tl.store(img_ptr + offset + 0, out_r, mask=mask)
    tl.store(img_ptr + offset + 1, out_g, mask=mask)
    tl.store(img_ptr + offset + 2, out_b, mask=mask)

@triton.jit
def tonemap_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(src_ptr + offs, mask=m, other=0.0)
    v = tl.minimum(tl.maximum(v, 0.0), 1.0) * 255.0
    tl.store(dst_ptr + offs, v.to(tl.uint8), mask=m)

# OpenGL helpers
VERT = """
#version 330 core
layout(location=0) in vec2 aPos;layout(location=1) in vec2 aUV;out vec2 vUV;void main(){vUV=aUV;gl_Position=vec4(aPos,0,1);}"""
FRAG = """
#version 330 core
in vec2 vUV;out vec4 FragColor;uniform sampler2D tex0;void main(){FragColor=texture(tex0,vUV);}"""

def compile_shader(src, stype):
    sid = gl.glCreateShader(stype)
    gl.glShaderSource(sid, src)
    gl.glCompileShader(sid)
    if gl.glGetShaderiv(sid, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
        raise RuntimeError(gl.glGetShaderInfoLog(sid).decode())
    return sid

def create_program():
    vs = compile_shader(VERT, gl.GL_VERTEX_SHADER)
    fs = compile_shader(FRAG, gl.GL_FRAGMENT_SHADER)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs); gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)
    if gl.glGetProgramiv(prog, gl.GL_LINK_STATUS) != gl.GL_TRUE:
        raise RuntimeError(gl.glGetProgramInfoLog(prog).decode())
    gl.glDeleteShader(vs); gl.glDeleteShader(fs)
    return prog

def create_fullscreen_quad():
    import numpy as np
    data = np.array([
        -1,-1, 0,0,
         1,-1, 1,0,
         1, 1, 1,1,
        -1, 1, 0,1
    ], dtype='f4')
    idx = np.array([0,1,2, 0,2,3], dtype='u4')
    vao = gl.glGenVertexArrays(1); gl.glBindVertexArray(vao)
    vbo = gl.glGenBuffers(1); gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
    ebo = gl.glGenBuffers(1); gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, gl.GL_STATIC_DRAW)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 16, ctypes.c_void_p(0))
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, False, 16, ctypes.c_void_p(8))
    return vao

def init_glfw():
    if not glfw.init():
        raise RuntimeError('glfw init failed')
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,3)
    glfw.window_hint(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(W, H, 'Triton Ray (GL)', None, None)
    if not win:
        raise RuntimeError('window create failed')
    glfw.make_context_current(win)
    return win

# CUDA-OpenGL interop via cuda-python
try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as drv
    import pycuda.gl as cuda_gl
    HAS_PYCUDA = True
except Exception:
    HAS_PYCUDA = False

if HAS_PYCUDA:
    class CudaPBO:
        def __init__(self, width, height):
            self.w = width; self.h = height
            self.pbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
            gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, width*height*3, None, gl.GL_STREAM_DRAW)
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
            self.res = cuda_gl.RegisteredBuffer(int(self.pbo), drv.gl.graphics_map_flags.WRITE_DISCARD)
        def map(self):
            self.res.map()
            ptr, size = self.res.device_ptr_and_size()
            return ptr, size
        def unmap(self):
            self.res.unmap()
        def delete(self):
            self.unmap()
            gl.glDeleteBuffers(1, [self.pbo])
else:
    class CudaPBO:
        def __init__(self, width, height):
            # Dummy; we'll use standard texture upload
            self.w = width; self.h = height
            self.pbo = None
        def map(self):
            return None, 0
        def unmap(self):
            pass
        def delete(self):
            pass

# Create GL resources
win = init_glfw()
program = create_program()
vao = create_fullscreen_quad()
tex = gl.glGenTextures(1)
gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
gl.glTexImage2D(gl.GL_TEXTURE_2D,0,gl.GL_RGB8,W,H,0,gl.GL_RGB,gl.GL_UNSIGNED_BYTE,None)
gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

# Interop PBO if available, else None
pbo = CudaPBO(W, H)

# Buffers
N_PIXELS = W * H
N_FLOATS = N_PIXELS * 3
float_img = torch.empty(N_FLOATS, device='cuda', dtype=torch.float32)
uint8_tmp = torch.empty(N_FLOATS, device='cuda', dtype=torch.uint8)
host_stage = None
if not HAS_PYCUDA:
    # Double-buffered pinned host staging
    host_stage = [torch.empty((H, W, 3), dtype=torch.uint8, pin_memory=True),
                  torch.empty((H, W, 3), dtype=torch.uint8, pin_memory=True)]
    copy_stream = torch.cuda.Stream()
    copy_done = [torch.cuda.Event(enable_timing=False), torch.cuda.Event(enable_timing=False)]

conv_grid = ((N_FLOATS + CONV_BLOCK - 1)//CONV_BLOCK,)
rt_grid = ((N_PIXELS + BLOCK - 1)//BLOCK,)

eye_buf = eye.to(torch.float32)
light_buf = light.to(torch.float32)

start_ev = torch.cuda.Event(enable_timing=True)
end_ev = torch.cuda.Event(enable_timing=True)
frames = 0
wall_start = time.time()
gpu_times = []
GPU_WINDOW = 120

try:
    while not glfw.window_should_close(win):
        glfw.poll_events()
        t_scalar = frames * 0.04
        t_sin = math.sin(t_scalar)
        t_cos = math.cos(t_scalar)
        start_ev.record()
        raytrace_kernel[rt_grid](float_img,
                                 eye_buf, light_buf,
                                 sphere_x, sphere_y, sphere_z, sphere_r,
                                 sphere_reflect, sphere_shine,
                                 t_sin, t_cos, NUM_SPHERES,
                                 H, W, BLOCK)
        tonemap_kernel[conv_grid](float_img, uint8_tmp, N_FLOATS, CONV_BLOCK)
        buf = frames & 1
        prev = buf ^ 1
        if HAS_PYCUDA and pbo.pbo is not None:
            # Map PBO and device->device copy
            ptr, size = pbo.map()
            drv.memcpy_dtod(ptr, uint8_tmp.data_ptr(), size)
            pbo.unmap()
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo.pbo)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D,0,0,0,W,H,gl.GL_RGB,gl.GL_UNSIGNED_BYTE,ctypes.c_void_p(0))
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        else:
            # Async copy to pinned buffer on copy_stream
            with torch.cuda.stream(copy_stream):
                host_stage[buf].view(-1).copy_(uint8_tmp, non_blocking=True)
                copy_done[buf].record(copy_stream)
            # Upload previous finished buffer
            if frames > 0:
                copy_done[prev].synchronize()
                gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
                # Direct pointer from pinned tensor
                ptr_prev = ctypes.c_void_p(host_stage[prev].data_ptr())
                gl.glTexSubImage2D(gl.GL_TEXTURE_2D,0,0,0,W,H,gl.GL_RGB,gl.GL_UNSIGNED_BYTE, ptr_prev)
        end_ev.record()
        # Draw fullscreen quad
        gl.glUseProgram(program)
        gl.glBindVertexArray(vao)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glfw.swap_buffers(win)
        frames += 1
        if frames and frames % 120 == 0:
            end_ev.synchronize()
            gpu_ms = start_ev.elapsed_time(end_ev)
            gpu_times.append(gpu_ms)
            if len(gpu_times) > GPU_WINDOW:
                gpu_times.pop(0)
            avg_gpu = sum(gpu_times)/len(gpu_times)
            gpu_fps = 1000.0/avg_gpu if avg_gpu > 0 else 0.0
            wall_fps = frames/(time.time()-wall_start)
            print(f"[Frame {frames}] gpu_avg {avg_gpu:.3f} ms (~{gpu_fps:.1f} FPS) display ~{wall_fps:.1f} FPS")
finally:
    glfw.terminate()
