"""
LingBot-World Gradio app -- multi-GPU (2x A6000) with FSDP + sequence parallelism.

Launch with:
    torchrun --nproc_per_node=2 gradio_app.py --port 7860
"""
import argparse
import logging
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import save_video

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [rank %(process)d] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)

EXAMPLES = {
    "00": "examples/00",
    "01": "examples/01",
    "02": "examples/02",
}

PROMPTS = {}
for eid, epath in EXAMPLES.items():
    prompt_file = os.path.join(epath, "prompt.txt")
    if os.path.exists(prompt_file):
        with open(prompt_file) as f:
            PROMPTS[eid] = f.read().strip()

SENTINEL_SHUTDOWN = "__SHUTDOWN__"

pipeline = None
rank = 0
world_size = 1


def setup_distributed():
    global rank, world_size
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=world_size)
        init_distributed_group()

    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    return local_rank


def load_pipeline(ckpt_dir: str, local_rank: int):
    global pipeline
    cfg = WAN_CONFIGS["i2v-A14B"]
    use_sp = world_size > 1
    dit_fsdp = world_size > 1

    logging.info(
        f"Loading pipeline: world_size={world_size}, "
        f"dit_fsdp={dit_fsdp}, t5_fsdp=False, t5_cpu=True, ulysses_size={world_size}"
    )
    pipeline = wan.WanI2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=dit_fsdp,
        use_sp=use_sp,
        t5_cpu=True,
    )
    logging.info("Pipeline loaded.")


def _broadcast_params(params=None):
    """Rank 0 sends generation params; other ranks receive them."""
    obj_list = [params] if rank == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def _run_generation(params):
    """All ranks call this together. Only rank 0 returns the video tensor."""
    prompt = params["prompt"]
    image_bytes = params["image_bytes"]
    action_path = params["action_path"]
    frame_num = params["frame_num"]
    seed = params["seed"]
    max_area = params["max_area"]

    img = None
    if image_bytes is not None:
        import io
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    video = pipeline.generate(
        input_prompt=prompt,
        img=img,
        action_path=action_path,
        max_area=max_area,
        frame_num=frame_num,
        seed=seed,
        offload_model=False,
    )
    return video


def generate_video(image, prompt, example_id, frame_num, seed):
    """Called by Gradio on rank 0 only."""
    if pipeline is None:
        return None, "Pipeline not loaded yet."
    if image is None:
        return None, "Please upload an image."
    if not prompt or not prompt.strip():
        return None, "Please enter a prompt."

    import io
    img_pil = Image.fromarray(image).convert("RGB")
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    action_path = EXAMPLES.get(example_id) if example_id != "none" else None

    frame_num = int(frame_num)
    if (frame_num - 1) % 4 != 0:
        frame_num = ((frame_num - 1) // 4) * 4 + 1
    frame_num = max(17, min(frame_num, 161))

    seed = int(seed) if seed >= 0 else 42
    max_area = MAX_AREA_CONFIGS["480*832"]

    params = {
        "prompt": prompt,
        "image_bytes": image_bytes,
        "action_path": action_path,
        "frame_num": frame_num,
        "seed": seed,
        "max_area": max_area,
    }

    logging.info(
        f"Generating: frames={frame_num}, seed={seed}, "
        f"action_path={action_path}, max_area={max_area}"
    )

    if world_size > 1:
        _broadcast_params(params)

    video = _run_generation(params)

    if video is None:
        return None, "Generation returned None (expected on non-rank-0)."

    cfg = WAN_CONFIGS["i2v-A14B"]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    save_video(
        tensor=video[None],
        save_file=tmp.name,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    logging.info(f"Video saved to {tmp.name}")
    del video
    torch.cuda.empty_cache()
    return tmp.name, "Done!"


def on_example_change(example_id):
    if example_id == "none":
        return None, ""
    epath = EXAMPLES.get(example_id, "")
    img_path = os.path.join(epath, "image.jpg")
    prompt = PROMPTS.get(example_id, "")
    img = Image.open(img_path).convert("RGB") if os.path.exists(img_path) else None
    return img, prompt


def worker_loop():
    """Non-rank-0 processes sit here waiting for generation requests."""
    logging.info(f"Worker rank {rank} entering loop.")
    while True:
        params = _broadcast_params()
        if params == SENTINEL_SHUTDOWN:
            break
        _run_generation(params)
    logging.info(f"Worker rank {rank} exiting.")


def build_and_launch(port, share):
    import gradio as gr

    with gr.Blocks(title="LingBot-World") as demo:
        gr.Markdown("# LingBot-World -- Interactive World Model")
        gr.Markdown(
            f"480P inference on {world_size} GPU(s) with FSDP + sequence parallelism."
        )

        with gr.Row():
            with gr.Column(scale=1):
                example_dropdown = gr.Dropdown(
                    choices=["none", "00", "01", "02"],
                    value="none",
                    label="Load Example",
                )
                image_input = gr.Image(label="Input Image", type="numpy")
                prompt_input = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    placeholder="Describe the video you want to generate...",
                )
                frame_slider = gr.Slider(
                    minimum=17, maximum=161, step=4, value=81,
                    label="Frame Count (4n+1, 16fps)",
                )
                seed_input = gr.Number(value=42, label="Seed (-1 for random)")
                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                video_output = gr.Video(label="Generated Video")
                status_output = gr.Textbox(label="Status", interactive=False)

        example_dropdown.change(
            fn=on_example_change,
            inputs=[example_dropdown],
            outputs=[image_input, prompt_input],
        )
        generate_btn.click(
            fn=generate_video,
            inputs=[image_input, prompt_input, example_dropdown,
                    frame_slider, seed_input],
            outputs=[video_output, status_output],
        )

    demo.launch(server_name="0.0.0.0", server_port=port, share=share)

    if world_size > 1:
        _broadcast_params(SENTINEL_SHUTDOWN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="./lingbot-world-base-cam")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=False)
    args = parser.parse_args()

    local_rank = setup_distributed()
    load_pipeline(args.ckpt_dir, local_rank)

    if rank == 0:
        build_and_launch(args.port, args.share)
    else:
        worker_loop()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
