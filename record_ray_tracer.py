from PIL import Image
import time
import math
import argparse
import subprocess
import shutil
import os
import sys
import tempfile
from tqdm import tqdm
import numpy as np
import importlib

# Optional OpenEXR imports; only required when writing EXR frames
try:
    import OpenEXR
    import Imath
    _HAS_OPENEXR = True
except Exception:
    _HAS_OPENEXR = False

# This script is derived from ray-tracer-1.py and records frames to a video file.

# Usage notes:
# - This repository provides a virtualenv at `.venv`. To run the script using that
#   environment use the Python executable at `.venv/bin/python`, for example:
#
#   .venv/bin/python record_ray_tracer.py --width 320 --height 240 --frames 600 \
#       --out out_video.mp4 --fps 30 --intro-text "My Demo" --intro-duration 3.0

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=1200, help="Number of frames to render (default 1200 -> 20s at 60fps)")
parser.add_argument("--out", type=str, default="out_video.mp4", help="Output video filename")
parser.add_argument("--fps", type=int, default=60, help="Frames per second")
parser.add_argument("--width", type=int, default=1024, help="Frame width")
parser.add_argument("--height", type=int, default=1024, help="Frame height")
parser.add_argument("--music", type=str, default=None, help="Path to music/audio file to add as an audio track")
parser.add_argument("--loop-music", action="store_true", help="Loop the music to match video duration when muxing")
parser.add_argument("--video-fade-in", type=float, default=0.0, help="Video fade in duration in seconds")
parser.add_argument("--video-fade-out", type=float, default=0.0, help="Video fade out duration in seconds")
parser.add_argument("--audio-fade-in", type=float, default=0.0, help="Audio fade in duration in seconds")
parser.add_argument("--audio-fade-out", type=float, default=0.0, help="Audio fade out duration in seconds")
parser.add_argument("--frames-out-dir", type=str, default=None, help="If set, write individual image frames to this directory instead of encoding a video with ffmpeg. Useful for importing into Blender.")
parser.add_argument("--frames-format", type=str, default="png", choices=["png", "exr"], help="Image format for raw frames (png or exr). PNG preserves 8-bit; EXR would preserve float HDR if supported.)")
parser.add_argument("--raytracer-module", type=str, default="ray_tracer", help="Name of the Python module containing the render_frame function (default: ray_tracer)")
parser.add_argument("--hdri-path", type=str, default=None, help="Path to HDRI EXR file to use for environment mapping")
parser.add_argument("--hdri-flip-y", dest='hdri_flip_y', action='store_true', help="Flip HDRI Y axis (default: true)")
parser.add_argument("--no-hdri-flip-y", dest='hdri_flip_y', action='store_false', help="Do not flip HDRI Y axis")
parser.set_defaults(hdri_flip_y=True)
args = parser.parse_args()

# Dynamically import the raytracer module
try:
    raytracer_module = importlib.import_module(args.raytracer_module)
    render_frame = raytracer_module.render_frame
except ImportError as e:
    print(f"Failed to import raytracer module '{args.raytracer_module}': {e}")
    sys.exit(1)
except AttributeError:
    print(f"Module '{args.raytracer_module}' does not have a 'render_frame' function")
    sys.exit(1)

# If the raytracer module supports HDRI configuration, pass the CLI options
if hasattr(raytracer_module, 'set_hdri'):
    try:
        if args.hdri_path is not None:
            raytracer_module.set_hdri(args.hdri_path, flip_y=args.hdri_flip_y)
        else:
            # If no path specified, at least set the flip flag if supported via module global
            try:
                setattr(raytracer_module, '_HDRI_FLIP_Y', args.hdri_flip_y)
            except Exception:
                pass
    except Exception as e:
        print(f"Warning: failed to configure HDRI on module '{args.raytracer_module}': {e}")

H, W = args.height, args.width
FRAMES = args.frames
FPS = args.fps
OUT_FILE = args.out
# Force output duration to exactly FRAMES / FPS seconds (prevents ffmpeg from
# extending the video to match a longer audio track). This is passed to ffmpeg
# as the '-t' output option.
DURATION_SECONDS_STR = f"{FRAMES / FPS:.6f}"
DURATION_SECONDS = FRAMES / FPS

# Ensure the parent directory for the output file exists. If the user supplied
# a path containing directories that don't yet exist, create them so ffmpeg
# (or OpenCV) won't fail later when opening/writing the file. If creation
# fails, exit with a helpful error message.
out_parent = os.path.dirname(OUT_FILE)
if out_parent:
    try:
        os.makedirs(out_parent, exist_ok=True)
    except Exception as e:
        print(f"Failed to create parent directory for output file '{out_parent}': {e}")
        sys.exit(1)



use_ffmpeg = shutil.which("ffmpeg") is not None
ffmpeg_proc = None
music_path = args.music
loop_music = args.loop_music
frames_out_dir = args.frames_out_dir
frames_format = args.frames_format

# Validate fade durations
if args.video_fade_in + args.video_fade_out > DURATION_SECONDS:
    print("Video fade in and out durations combined cannot exceed video duration")
    sys.exit(1)
if music_path is not None and args.audio_fade_in + args.audio_fade_out > DURATION_SECONDS:
    print("Audio fade in and out durations combined cannot exceed video duration")
    sys.exit(1)

if frames_out_dir is None and not use_ffmpeg:
    print("This script requires ffmpeg to write video. Please install ffmpeg, or use --frames-out-dir to write raw frames instead.")
    sys.exit(1)

# Sanity-check the music file early so we fail fast if path is invalid
if music_path is not None:
    if not os.path.exists(music_path):
        print(f"Music file '{music_path}' does not exist.")
        sys.exit(1)
    if not os.path.isfile(music_path):
        print(f"Music path '{music_path}' is not a file.")
        sys.exit(1)
    # Probe the audio file with ffprobe (preferred) or ffmpeg to ensure it's decodable
    if shutil.which('ffprobe') is not None:
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', music_path]
        try:
            subprocess.run(probe_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"ffprobe failed to parse audio file '{music_path}'. Please provide a valid audio file.")
            sys.exit(1)
    elif shutil.which('ffmpeg') is not None:
        # fallback: try to decode the file (no output) to ensure ffmpeg can read it
        probe_cmd = ['ffmpeg', '-v', 'error', '-i', music_path, '-f', 'null', '-']
        try:
            subprocess.run(probe_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"ffmpeg failed to decode audio file '{music_path}'. Please provide a valid audio file.")
            sys.exit(1)
    else:
        # No ffmpeg/ffprobe available; warn but continue — muxing will fail later if attempted
        tqdm.write("Warning: ffprobe/ffmpeg not found; cannot verify audio file before rendering.")

# Spawn ffmpeg to read raw BGR24 frames from stdin and encode to mp4
ffmpeg_proc = None

# If frames_out_dir is set, we'll write individual image files and skip spawning ffmpeg
if frames_out_dir is None:
    ffmpeg_cmd = ["ffmpeg", '-y']
    # Video input from stdin
    ffmpeg_cmd += ['-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f"{W}x{H}", '-r', str(FPS), '-i', '-']
    if music_path is not None:
        # If music provided, add it as a second input. If looping desired, use -stream_loop.
        if loop_music:
            # Use -stream_loop before the input to loop the audio indefinitely (-1 loops forever)
            ffmpeg_cmd += ['-stream_loop', '-1']
        ffmpeg_cmd += ['-i', music_path]
        # Map video from first input and audio from second, set audio codec
        ffmpeg_cmd += ['-map', '0:v:0', '-map', '1:a:0', '-c:a', 'aac', '-b:a', '192k']
    else:
        ffmpeg_cmd += ['-an']
    # Add fade filters
    if args.video_fade_in > 0 or args.video_fade_out > 0:
        vf_parts = []
        if args.video_fade_in > 0:
            vf_parts.append(f"fade=t=in:st=0:d={args.video_fade_in}")
        if args.video_fade_out > 0:
            vf_parts.append(f"fade=t=out:st={DURATION_SECONDS - args.video_fade_out}:d={args.video_fade_out}")
        ffmpeg_cmd += ['-filter:v', ','.join(vf_parts)]
    if music_path is not None and (args.audio_fade_in > 0 or args.audio_fade_out > 0):
        af_parts = []
        if args.audio_fade_in > 0:
            af_parts.append(f"afade=t=in:st=0:d={args.audio_fade_in}")
        if args.audio_fade_out > 0:
            af_parts.append(f"afade=t=out:st={DURATION_SECONDS - args.audio_fade_out}:d={args.audio_fade_out}")
        ffmpeg_cmd += ['-filter:a', ','.join(af_parts)]
    # Ensure ffmpeg produces a file with the exact requested duration.
    ffmpeg_cmd += ['-t', DURATION_SECONDS_STR]
    ffmpeg_cmd += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', OUT_FILE]
    # Silence ffmpeg's console output by redirecting stdout/stderr to DEVNULL
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)

# If requested, create frames output directory
if frames_out_dir is not None:
    try:
        os.makedirs(frames_out_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create frames output directory '{frames_out_dir}': {e}")
        sys.exit(1)

tqdm.write(f"Rendering {FRAMES} frames ({FRAMES/FPS:.1f}s) to {OUT_FILE} at {FPS} FPS ({W}x{H})")

start_time = time.time()
frame_time_start = time.time()

try:
    iterator = tqdm(range(FRAMES), desc="Rendering", unit="frame", dynamic_ncols=True)
    for frame in iterator:
        frame_rgb = render_frame(frame, W, H).numpy()
        if frames_out_dir is not None:
            # Save frame as an image file suitable for Blender import
            pad = max(6, len(str(FRAMES)))
            filename = os.path.join(frames_out_dir, f"frame_{frame:0{pad}d}.{frames_format}")
            if frames_format == 'png':
                img = Image.fromarray(frame_rgb)
                try:
                    img.save(filename)
                except Exception as e:
                    tqdm.write(f"Failed to save frame {frame} to '{filename}': {e}")
                    break
            else:
                # EXR export
                if not _HAS_OPENEXR:
                    tqdm.write("OpenEXR/Imath not installed; cannot write EXR files. Please install OpenEXR and Imath (added to requirements.txt).")
                    break
                try:
                    # Convert uint8 RGB (H,W,3) to float32 linear RGB in range [0,1]
                    arr = np.asarray(frame_rgb, dtype=np.uint8)
                    # Convert to float32 and normalize
                    f = (arr.astype(np.float32) / 255.0)
                    # Flip to channel-first order and convert to bytes
                    R = (f[..., 0].astype(np.float32)).tobytes()
                    G = (f[..., 1].astype(np.float32)).tobytes()
                    B = (f[..., 2].astype(np.float32)).tobytes()

                    header = OpenEXR.Header(W, H)
                    pt = Imath.PixelType(Imath.PixelType.FLOAT)
                    header['channels'] = dict(R=Imath.Channel(pt), G=Imath.Channel(pt), B=Imath.Channel(pt))
                    exr = OpenEXR.OutputFile(filename, header)
                    exr.writePixels({'R': R, 'G': G, 'B': B})
                    exr.close()
                except Exception as e:
                    tqdm.write(f"Failed to write EXR frame {frame} to '{filename}': {e}")
                    break
        else:
            frame_bgr = frame_rgb[..., ::-1]
            if ffmpeg_proc is not None:
                # write raw bytes to ffmpeg stdin
                try:
                    ffmpeg_proc.stdin.write(frame_bgr.tobytes())
                except BrokenPipeError:
                    tqdm.write("ffmpeg pipe closed unexpectedly")
                    break
        
        # Update tqdm postfix with average FPS
        iterator.set_postfix({'avg_fps': f"{(frame+1)/max(1e-6, time.time()-start_time):.2f}"})

except Exception as e:
    print("Error during rendering:", e)
finally:
    if ffmpeg_proc is not None:
        try:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
        except Exception:
            pass
    total = time.time() - start_time
    tqdm.write(f"Done: {FRAMES} frames in {total:.2f}s => {FRAMES/total:.2f} FPS (overall)")
