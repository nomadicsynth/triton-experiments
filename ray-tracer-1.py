import triton
import triton.language as tl
import torch
from PIL import Image
import time
import math
import cv2

# Image size
H, W = 512, 512

# Camera
eye = torch.tensor([0.0, 0.0, -3.0], device="cuda")
light = torch.tensor([5.0, 5.0, -5.0], device="cuda")

# Base (rest) sphere data (x,y,z,r)
spheres_base = torch.tensor(
    [[0.0, 0.0, 0.0, 0.5],
     [1.0, 0.0, 0.0, 0.5]],
    device="cuda"
).to(torch.float32)
spheres_buf = spheres_base.clone()


@triton.jit
def raytrace_kernel(img_ptr, spheres_ptr, eye_ptr, light_ptr, H: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(0)
    y = pid // W
    x = pid % W
    if (y >= H) | (x >= W):
        return

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

    color_r = 0.0
    color_g = 0.0
    color_b = 0.0
    closest_t = 1e9

    # Two spheres (compile-time unrolled)
    for i in range(2):
        base = i * 4
        s_x = tl.load(spheres_ptr + base + 0)
        s_y = tl.load(spheres_ptr + base + 1)
        s_z = tl.load(spheres_ptr + base + 2)
        s_r = tl.load(spheres_ptr + base + 3)

        oc_x = eye_x - s_x
        oc_y = eye_y - s_y
        oc_z = eye_z - s_z

        b = 2.0 * (oc_x * rd_x + oc_y * rd_y + oc_z * rd_z)
        c = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z - s_r * s_r
        disc = b * b - 4.0 * c
        if disc > 0:
            sqrt_disc = tl.sqrt(disc)
            t = (-b - sqrt_disc) * 0.5
            if (t > 0) & (t < closest_t):
                closest_t = t
                p_x = eye_x + rd_x * t
                p_y = eye_y + rd_y * t
                p_z = eye_z + rd_z * t
                n_x = (p_x - s_x) / s_r
                n_y = (p_y - s_y) / s_r
                n_z = (p_z - s_z) / s_r
                l_x = light_x - p_x
                l_y = light_y - p_y
                l_z = light_z - p_z
                l_inv_len = 1.0 / tl.sqrt(l_x * l_x + l_y * l_y + l_z * l_z)
                l_x *= l_inv_len
                l_y *= l_inv_len
                l_z *= l_inv_len
                diff = n_x * l_x + n_y * l_y + n_z * l_z
                diff = tl.maximum(diff, 0.0)
                color_r = diff
                color_g = diff
                color_b = diff

    offset = pid * 3
    tl.store(img_ptr + offset + 0, color_r)
    tl.store(img_ptr + offset + 1, color_g)
    tl.store(img_ptr + offset + 2, color_b)


# Allocate once
img = torch.zeros(H * W * 3, device="cuda", dtype=torch.float32)
eye_buf = eye.to(torch.float32)
light_buf = light.to(torch.float32)

grid = (H * W,)

cv2.namedWindow("ray", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ray", W, H)

save_first_frame = True
frame = 0
t0 = time.time()
fps_avg_window = 30
times = []
try:
    while True:
        t = frame * 0.04
        # Animate sphere x positions (host update)
        # Avoid reallocations; write in-place
        spheres_buf[0, 0] = math.sin(t) * 0.5
        spheres_buf[1, 0] = 1.0 + math.cos(t) * 0.5

        torch.cuda.synchronize()
        start = time.time()
        raytrace_kernel[grid](
            img,
            spheres_buf,
            eye_buf,
            light_buf,
            H,
            W
        )
        torch.cuda.synchronize()
        render_ms = (time.time() - start) * 1000.0

        # Copy to CPU & update display
        frame_img = (img.view(H, W, 3).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        # Convert RGB (ours) -> BGR (OpenCV)
        frame_img_bgr = frame_img[..., ::-1]
        cv2.imshow("ray", frame_img_bgr)

        if save_first_frame and frame == 0:
            Image.fromarray(frame_img).save("out.png")
            print("Saved first frame to out.png")
            save_first_frame = False

        times.append(render_ms)
        if len(times) > fps_avg_window:
            times.pop(0)
        if frame % 30 == 0 and frame > 0:
            avg_ms = sum(times) / len(times)
            print(f"[Frame {frame}] avg {avg_ms:.2f} ms  ~ {1000.0/avg_ms:.1f} FPS")

        # Small wait; capture 'q' key to exit. 1ms keeps UI responsive.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame += 1
except KeyboardInterrupt:
    pass
finally:
    total = time.time() - t0
    if frame > 0:
        print(f"Stopped. {frame} frames in {total:.2f}s => {frame/total:.2f} FPS avg.")
    cv2.destroyAllWindows()
