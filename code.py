import os
import sys
import math
import json
import tempfile
import asyncio
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from nudenet import NudeDetector
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ==================== LOGGING ====================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("NSFWBot")
logging.getLogger("httpx").setLevel(logging.WARNING)

# ==================== CONFIGURATION ====================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
SERVER_URL = (
    os.getenv("SERVER_URL") or os.getenv("RENDER_EXTERNAL_URL") or ""
).strip().rstrip("/")
PORT = int(os.getenv("PORT", 10000))

ADD_GROUP_URL = "https://t.me/Smart_Nsfw_Detector_Bot?startgroup=true"
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8 MB limit
TARGET_DIMENSION = 320           # 320x320 compressed & padded canvas
MAX_VIDEO_FRAMES = 10            # Max 10 frames per video at constant time gaps
LIVE_STICKER_FPS = 1             # 1 FPS sampling for live stickers
CACHE_MAX_ITEMS = 2500           # In-memory cache for repeated media

# Per-class sensitivity thresholds (triggers deletion when score >= class limit)
CLASS_THRESHOLDS = {
    # Primary Explicit Classes (Strict Limits)
    "FEMALE_GENITALIA_EXPOSED": 0.55,
    "MALE_GENITALIA_EXPOSED": 0.55,
    "ANUS_EXPOSED": 0.55,
    "FEMALE_BREAST_EXPOSED": 0.60,
    "BUTTOCKS_EXPOSED": 0.65,
    # Additional Explicit & Borderline NSFW Classes (Balanced Limits)
    "MALE_BREAST_EXPOSED": 0.72,
    "FEMALE_GENITALIA_COVERED": 0.70,
    "ANUS_COVERED": 0.70,
    "FEMALE_BREAST_COVERED": 0.78,
    "BUTTOCKS_COVERED": 0.78,
}

if not BOT_TOKEN:
    logger.error("BOT_TOKEN environment variable is missing!")
    sys.exit(1)

# ==================== DETECTOR, QUEUES & CACHE ====================
# Single-worker thread pool ensures strictly one-by-one synchronous model execution
detector = NudeDetector()
scan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="SingleScanner")

webhook_raw_queue: asyncio.Queue = asyncio.Queue()
scan_queue: asyncio.Queue = asyncio.Queue()
download_semaphore = asyncio.Semaphore(3)

# LRU Cache: {file_unique_id: is_nsfw_bool}
result_cache: OrderedDict[str, bool] = OrderedDict()


def get_cached_result(unique_id: str) -> bool | None:
    if not unique_id or unique_id not in result_cache:
        return None
    result_cache.move_to_end(unique_id)
    return result_cache[unique_id]


def set_cached_result(unique_id: str, is_nsfw: bool) -> None:
    if not unique_id:
        return
    result_cache[unique_id] = is_nsfw
    result_cache.move_to_end(unique_id)
    if len(result_cache) > CACHE_MAX_ITEMS:
        result_cache.popitem(last=False)


# ==================== BLUE BUTTON HELPER ====================
def get_add_to_group_markup() -> InlineKeyboardMarkup:
    """Creates a solid blue Telegram button redirecting to the group add URL."""
    try:
        button = InlineKeyboardButton(
            text="➕ Add to Your Group",
            url=ADD_GROUP_URL,
            style="primary",
        )
    except TypeError:
        button = InlineKeyboardButton(
            text="➕ Add to Your Group",
            url=ADD_GROUP_URL,
            api_kwargs={"style": "primary"},
        )
    return InlineKeyboardMarkup([[button]])


# ==================== IMAGE & FRAME PREPROCESSING ====================
def normalize_channels(img: np.ndarray) -> np.ndarray:
    """Handles RGBA/BGRA transparency (common in stickers) and grayscale images."""
    if img is None or img.size == 0:
        return np.zeros((TARGET_DIMENSION, TARGET_DIMENSION, 3), dtype=np.uint8)

    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = (img[:, :, 3].astype(np.float32)) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        bg = np.full_like(bgr, 240.0)  # Neutral light background for transparent stickers
        composited = bgr * alpha[:, :, np.newaxis] + bg * (1.0 - alpha[:, :, np.newaxis])
        return composited.astype(np.uint8)

    return img[:, :, :3]


def compress_and_pad_frame(frame: np.ndarray, target_size: int = TARGET_DIMENSION) -> np.ndarray:
    """
    Compresses the frame and adds uniform border padding to fit a 320x320 canvas.
    Preserves aspect ratio and keeps edge features clearly visible for detection.
    """
    clean_frame = normalize_channels(frame)
    h, w = clean_frame.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    inner_size = int(target_size * 0.88)
    scale = min(inner_size / max(w, 1), inner_size / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(clean_frame, (new_w, new_h), interpolation=interp)

    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[114, 114, 114],  # Neutral padding color for best detection accuracy
    )


def is_frame_nsfw_sync(frame: np.ndarray) -> tuple[bool, str, float]:
    """
    Evaluates a single padded 320x320 frame against CLASS_THRESHOLDS.
    Returns (is_nsfw, matched_class, matched_score).
    """
    padded = compress_and_pad_frame(frame, TARGET_DIMENSION)
    detections = detector.detect(padded)

    for item in detections:
        label = item.get("class", "")
        score = float(item.get("score", 0.0))
        threshold = CLASS_THRESHOLDS.get(label)
        if threshold is not None and score >= threshold:
            return True, label, score

    return False, "", 0.0


# ==================== MEDIA SCANNERS (STATIC / LIVE STICKER / VIDEO) ====================
def scan_live_sticker_sync(file_path: str) -> bool:
    """
    Dedicated scanner for Live Video Stickers (.webm) at 1 FPS.
    Decodes all frames reliably (ignoring broken WebM header metadata) and samples
    at 1-second intervals plus key boundary frames (start, middle, end).
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return False

    frames: list[np.ndarray] = []
    timestamps_ms: list[float] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(frame)
            timestamps_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    finally:
        cap.release()

    total = len(frames)
    if total == 0:
        return False

    # Select frames at 1 FPS (every 1000ms)
    indices_to_scan: list[int] = [0]
    last_sampled_ms = timestamps_ms[0] if timestamps_ms[0] >= 0 else 0.0
    interval_ms = 1000.0 / LIVE_STICKER_FPS  # 1000ms for 1 FPS

    for idx in range(1, total):
        cur_ms = timestamps_ms[idx]
        if cur_ms > 0 and (cur_ms - last_sampled_ms) >= interval_ms:
            indices_to_scan.append(idx)
            last_sampled_ms = cur_ms

    # Fallback if container timestamps were missing or 0 (assume standard ~30fps live sticker)
    if len(indices_to_scan) == 1 and total > 1:
        step = max(1, int(round(30 / LIVE_STICKER_FPS)))
        indices_to_scan = list(range(0, total, step))

    # Always include middle and last frame so short bait stickers cannot bypass 1 FPS
    for extra_idx in (total // 3, (2 * total) // 3, total - 1):
        if 0 <= extra_idx < total and extra_idx not in indices_to_scan:
            indices_to_scan.append(extra_idx)

    for idx in sorted(set(indices_to_scan)):
        flagged, label, score = is_frame_nsfw_sync(frames[idx])
        if flagged:
            logger.info(f"Live sticker flagged on frame {idx + 1}/{total}: {label} ({score:.2f})")
            return True

    return False


def scan_video_10_frames_sync(file_path: str) -> bool:
    """
    Scans any video or GIF by extracting at most 10 frames at a constant time gap
    across the entire video duration. Exits early as soon as NSFW is detected.
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return False

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Path A: Reliable frame count known -> seek across 10 constant-gap positions
        if total_frames > 0:
            num_samples = min(MAX_VIDEO_FRAMES, total_frames)
            target_indices = np.linspace(0, total_frames - 1, num=num_samples, dtype=int)
            last_idx = -1

            for idx in target_indices:
                idx_int = int(idx)
                if idx_int == last_idx:
                    continue
                last_idx = idx_int

                cap.set(cv2.CAP_PROP_POS_FRAMES, idx_int)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                flagged, label, score = is_frame_nsfw_sync(frame)
                if flagged:
                    logger.info(
                        f"Video flagged at frame {idx_int + 1}/{total_frames}: {label} ({score:.2f})"
                    )
                    return True
            return False

        # Path B: Fallback when container doesn't report total_frames
        all_frames: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            all_frames.append(frame)

        total_read = len(all_frames)
        if total_read == 0:
            return False

        num_samples = min(MAX_VIDEO_FRAMES, total_read)
        target_indices = np.linspace(0, total_read - 1, num=num_samples, dtype=int)

        for idx in sorted(set(int(i) for i in target_indices)):
            flagged, label, score = is_frame_nsfw_sync(all_frames[idx])
            if flagged:
                logger.info(f"Video flagged at frame {idx + 1}/{total_read}: {label} ({score:.2f})")
                return True

        return False
    finally:
        cap.release()


def scan_media_sync(file_path: str, media_kind: str) -> bool:
    """
    Main synchronous entry point executed strictly one-by-one in the worker thread.
    media_kind: "image" | "live_sticker" | "video"
    """
    if media_kind == "live_sticker":
        return scan_live_sticker_sync(file_path)

    if media_kind == "video":
        return scan_video_10_frames_sync(file_path)

    # Static image or static sticker (.jpg, .png, .webp)
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        # Fallback if a sticker or image requires VideoCapture decoding
        return scan_live_sticker_sync(file_path)

    flagged, label, score = is_frame_nsfw_sync(img)
    if flagged:
        logger.info(f"Image flagged: {label} ({score:.2f})")
    return flagged


# ==================== TELEGRAM HANDLERS ====================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /start with a concise English message and blue Add to Group button."""
    if not update.effective_message:
        return

    text = (
        "🛡️ <b>Smart NSFW Detector Bot</b>\n\n"
        "<blockquote>I'm an advanced NSFW detector bot. I can delete NSFW content "
        "from your GC to keep your community clean and safe.</blockquote>\n\n"
        "📌 <i>Add me to your group and grant <b>Delete Messages</b> admin permission.</i>"
    )

    await update.effective_message.reply_text(
        text=text,
        parse_mode=ParseMode.HTML,
        reply_markup=get_add_to_group_markup(),
    )


def extract_media_info(message) -> tuple[str | None, str | None, int, str, str]:
    """
    Extracts (file_id, file_unique_id, file_size, media_kind, suffix).
    media_kind is one of: "image", "live_sticker", "video", or "".
    """
    # 1. Stickers (Static .webp, Live Video .webm, Animated .tgs)
    if message.sticker:
        st = message.sticker
        if st.is_video:
            return st.file_id, st.file_unique_id, st.file_size or 0, "live_sticker", ".webm"
        if st.is_animated:
            # Animated .tgs stickers: check embedded thumbnail
            if st.thumbnail:
                return (
                    st.thumbnail.file_id,
                    st.file_unique_id,
                    st.thumbnail.file_size or 0,
                    "image",
                    ".webp",
                )
            return None, None, 0, "", ""
        return st.file_id, st.file_unique_id, st.file_size or 0, "image", ".webp"

    # 2. GIFs / Animations
    if message.animation:
        anim = message.animation
        return anim.file_id, anim.file_unique_id, anim.file_size or 0, "video", ".mp4"

    # 3. Photos
    if message.photo:
        p = message.photo[-1]
        return p.file_id, p.file_unique_id, p.file_size or 0, "image", ".jpg"

    # 4. Videos
    if message.video:
        v = message.video
        return v.file_id, v.file_unique_id, v.file_size or 0, "video", ".mp4"

    # 5. Video Notes
    if message.video_note:
        vn = message.video_note
        return vn.file_id, vn.file_unique_id, vn.file_size or 0, "video", ".mp4"

    # 6. Documents (uncompressed images, GIFs, WebM, videos)
    if message.document:
        doc = message.document
        mime = (doc.mime_type or "").lower()
        if "webm" in mime:
            return doc.file_id, doc.file_unique_id, doc.file_size or 0, "live_sticker", ".webm"
        if mime.startswith("image/"):
            if "gif" in mime:
                return doc.file_id, doc.file_unique_id, doc.file_size or 0, "video", ".gif"
            return doc.file_id, doc.file_unique_id, doc.file_size or 0, "image", ".jpg"
        if mime.startswith("video/"):
            return doc.file_id, doc.file_unique_id, doc.file_size or 0, "video", ".mp4"

    return None, None, 0, "", ""


async def delete_and_notify(bot, chat_id: int, message_id: int, user_mention: str) -> None:
    """Deletes the flagged message and posts a formatted alert with the blue button."""
    try:
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception as e:
        logger.warning(f"Could not delete msg {message_id} in {chat_id}: {e}")
        return

    alert_text = (
        "🚫 <b>NSFW Content Deleted</b>\n\n"
        f"<blockquote>Deleted a message from {user_mention} for containing NSFW content.</blockquote>\n\n"
        "🛡️ <i>Add this bot to your GC to remove NSFW content.</i>"
    )
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=alert_text,
            parse_mode=ParseMode.HTML,
            reply_markup=get_add_to_group_markup(),
        )
    except Exception as e:
        logger.warning(f"Failed to send alert in {chat_id}: {e}")


async def send_size_limit_reply(bot, chat_id: int, message_id: int) -> None:
    """Replies when a file is larger than 8 MB."""
    limit_text = (
        "⚠️ <b>File Limit Exceeded</b>\n\n"
        "<blockquote>This file exceeds my <b>8 MB</b> limit, so I cannot process it.</blockquote>"
    )
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=limit_text,
            reply_to_message_id=message_id,
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        logger.warning(f"Failed to send limit notice in {chat_id}: {e}")


async def prefetch_and_enqueue(bot, job: dict) -> None:
    """Pre-downloads media in the background and queues it for one-by-one scanning."""
    async with download_semaphore:
        temp_path = None
        try:
            tg_file = await bot.get_file(job["file_id"])
            if tg_file.file_size and tg_file.file_size > MAX_FILE_SIZE:
                await send_size_limit_reply(bot, job["chat_id"], job["message_id"])
                return

            fd, temp_path = tempfile.mkstemp(suffix=job["suffix"])
            os.close(fd)
            await tg_file.download_to_drive(custom_path=temp_path)

            job["temp_path"] = temp_path
            await scan_queue.put(job)
        except Exception as e:
            logger.error(f"Download error for {job['file_id']}: {e}")
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


async def handle_media_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Validates media size, checks instant cache, or queues for sequential scan."""
    message = update.effective_message
    chat = update.effective_chat
    if not message or not chat:
        return

    file_id, unique_id, file_size, media_kind, suffix = extract_media_info(message)
    if not file_id or not media_kind:
        return

    if file_size > MAX_FILE_SIZE:
        await send_size_limit_reply(context.bot, chat.id, message.message_id)
        return

    user_mention = (
        message.from_user.mention_html() if message.from_user else "<b>User</b>"
    )

    # Check cache first for instant action on repeated stickers/GIFs
    cached_flag = get_cached_result(unique_id)
    if cached_flag is not None:
        if cached_flag:
            await delete_and_notify(context.bot, chat.id, message.message_id, user_mention)
        return

    job = {
        "chat_id": chat.id,
        "message_id": message.message_id,
        "user_mention": user_mention,
        "file_id": file_id,
        "unique_id": unique_id,
        "media_kind": media_kind,
        "suffix": suffix,
    }
    asyncio.create_task(prefetch_and_enqueue(context.bot, job))


# ==================== BACKGROUND WORKERS ====================
async def webhook_processor_worker(bot_app: Application) -> None:
    """Consumes raw webhook payloads from memory so /webhook responds in <1ms."""
    while True:
        raw_bytes = await webhook_raw_queue.get()
        try:
            data = json.loads(raw_bytes)
            update = Update.de_json(data, bot_app.bot)
            if update:
                asyncio.create_task(bot_app.process_update(update))
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Webhook worker error: {e}")
        finally:
            webhook_raw_queue.task_done()


async def sequential_scan_worker(bot_app: Application) -> None:
    """Executes NudeNet scans strictly one-by-one in the dedicated single thread."""
    loop = asyncio.get_running_loop()

    while True:
        job = await scan_queue.get()
        temp_path = job.get("temp_path")
        try:
            cached = get_cached_result(job["unique_id"])
            if cached is not None:
                is_nsfw = cached
            else:
                is_nsfw = await loop.run_in_executor(
                    scan_executor,
                    scan_media_sync,
                    temp_path,
                    job["media_kind"],
                )
                set_cached_result(job["unique_id"], is_nsfw)

            if is_nsfw:
                await delete_and_notify(
                    bot_app.bot,
                    job["chat_id"],
                    job["message_id"],
                    job["user_mention"],
                )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Sequential scan worker error: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            scan_queue.task_done()


async def self_ping_loop() -> None:
    """Pings the server URL every 30 seconds to prevent Render spin-down."""
    if not SERVER_URL:
        return

    ping_url = f"{SERVER_URL}/health"
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            await asyncio.sleep(30)
            try:
                await client.get(ping_url)
            except asyncio.CancelledError:
                break
            except Exception:
                pass


# ==================== FASTAPI & LIFESPAN ====================
t_request = HTTPXRequest(connection_pool_size=32, read_timeout=30.0, write_timeout=30.0)
ptb_app = Application.builder().token(BOT_TOKEN).request(t_request).updater(None).build()

MEDIA_FILTERS = (
    filters.Sticker.ALL
    | filters.ANIMATION
    | filters.PHOTO
    | filters.VIDEO
    | filters.VIDEO_NOTE
    | filters.Document.IMAGE
    | filters.Document.VIDEO
)

ptb_app.add_handler(CommandHandler("start", start_command))
ptb_app.add_handler(MessageHandler(MEDIA_FILTERS, handle_media_message))


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ptb_app.initialize()
    await ptb_app.start()

    if SERVER_URL:
        webhook_url = f"{SERVER_URL}/webhook"
        await ptb_app.bot.set_webhook(
            url=webhook_url,
            drop_pending_updates=True,
            allowed_updates=["message", "edited_message"],
        )
        logger.info(f"Webhook active at: {webhook_url}")

    webhook_task = asyncio.create_task(webhook_processor_worker(ptb_app))
    scan_task = asyncio.create_task(sequential_scan_worker(ptb_app))
    ping_task = asyncio.create_task(self_ping_loop())

    yield

    webhook_task.cancel()
    scan_task.cancel()
    ping_task.cancel()
    await ptb_app.stop()
    await ptb_app.shutdown()
    scan_executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)


@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/webhook")
async def telegram_webhook(request: Request):
    """Instantly queues raw webhook bytes and returns 200 OK without waiting."""
    body = await request.body()
    if body:
        webhook_raw_queue.put_nowait(body)
    return Response(status_code=200)


if __name__ == "__main__":
    uvicorn.run("code:app", host="0.0.0.0", port=PORT, log_level="info")
