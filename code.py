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
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8 MB
NSFW_THRESHOLD = 0.80            # 80% score threshold
TARGET_DIMENSION = 320           # 320x320 compressed & padded size
SCAN_FPS = 2                     # 2 FPS for videos/GIFs/live stickers
CACHE_MAX_ITEMS = 2000           # Cache size for repeated stickers/GIFs

NSFW_TARGET_LABELS = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
    "FEMALE_GENITALIA_COVERED",
    "ANUS_COVERED",
}

if not BOT_TOKEN:
    logger.error("BOT_TOKEN environment variable is missing!")
    sys.exit(1)

# ==================== DETECTOR, QUEUES & CACHE ====================
# Single-worker executor guarantees strictly one-by-one synchronous scanning
detector = NudeDetector()
scan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="SingleScanner")

# Separate webhook & scan queues so webhook returns 200 OK instantaneously
webhook_raw_queue: asyncio.Queue = asyncio.Queue()
scan_queue: asyncio.Queue = asyncio.Queue()
download_semaphore = asyncio.Semaphore(3)  # Pre-fetch up to 3 files ahead

# LRU Cache: {file_unique_id: max_score} to skip re-scanning repeated stickers/GIFs
score_cache: OrderedDict[str, float] = OrderedDict()


def get_cached_score(unique_id: str) -> float | None:
    if not unique_id or unique_id not in score_cache:
        return None
    score_cache.move_to_end(unique_id)
    return score_cache[unique_id]


def set_cached_score(unique_id: str, score: float) -> None:
    if not unique_id:
        return
    score_cache[unique_id] = score
    score_cache.move_to_end(unique_id)
    if len(score_cache) > CACHE_MAX_ITEMS:
        score_cache.popitem(last=False)


# ==================== BLUE BUTTON HELPER ====================
def get_add_to_group_markup() -> InlineKeyboardMarkup:
    """Returns a solid blue Telegram button redirecting to the group add link."""
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


# ==================== IMAGE & VIDEO PROCESSING ====================
def compress_and_pad_frame(frame: np.ndarray, target_size: int = TARGET_DIMENSION) -> np.ndarray:
    """Compresses frame into a padded 320x320 square canvas."""
    if frame is None or frame.size == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    inner_size = int(target_size * 0.85)
    h, w = frame.shape[:2]

    scale = min(inner_size / max(w, 1), inner_size / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

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
        value=[0, 0, 0],
    )


def evaluate_frame_sync(frame: np.ndarray) -> float:
    """Runs synchronous detection on a single padded 320x320 frame."""
    padded = compress_and_pad_frame(frame, TARGET_DIMENSION)
    detections = detector.detect(padded)

    max_score = 0.0
    for item in detections:
        if item.get("class") in NSFW_TARGET_LABELS:
            score = float(item.get("score", 0.0))
            if score > max_score:
                max_score = score
    return max_score


def scan_media_sync(file_path: str, is_video_or_anim: bool) -> float:
    """
    Runs strictly one-by-one inside the single worker thread.
    - Static images/stickers: compressed & padded to 320x320.
    - Live stickers/GIFs/videos: sampled at 2 FPS, compressed & padded to 320x320,
      and stops immediately as soon as any frame hits 80%+.
    """
    if not is_video_or_anim:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if len(img.shape) == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3] / 255.0
                bgr = img[:, :, :3]
                bg = np.full_like(bgr, 255, dtype=np.uint8)
                img = (bgr * alpha[:, :, np.newaxis] + bg * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return evaluate_frame_sync(img)

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return 0.0

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or math.isnan(native_fps) or native_fps <= 0 or native_fps > 120:
        native_fps = 30.0

    frame_step = max(1, int(round(native_fps / SCAN_FPS)))
    frame_idx = 0
    highest_score = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                score = evaluate_frame_sync(frame)
                if score > highest_score:
                    highest_score = score
                if highest_score >= NSFW_THRESHOLD:
                    break  # Instant early exit on NSFW detection

            frame_idx += 1
    finally:
        cap.release()

    return highest_score


# ==================== TELEGRAM HANDLERS ====================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a clean, concise /start response with the blue Add to Group button."""
    if not update.effective_message:
        return

    text = (
        "🛡️ <b>Smart NSFW Detector Bot</b>\n\n"
        "<blockquote>I'm an advanced NSFW detector bot. I can delete NSFW content "
        "from your GC to keep your chat clean and safe.</blockquote>\n\n"
        "📌 <i>Add me to your group with <b>Delete Messages</b> admin permission.</i>"
    )

    await update.effective_message.reply_text(
        text=text,
        parse_mode=ParseMode.HTML,
        reply_markup=get_add_to_group_markup(),
    )


def extract_media_info(message) -> tuple[str | None, str | None, int, bool, str]:
    """Returns (file_id, file_unique_id, file_size, is_video_or_anim, suffix)."""
    if message.sticker:
        st = message.sticker
        if st.is_video:
            return st.file_id, st.file_unique_id, st.file_size or 0, True, ".webm"
        if st.is_animated:
            if st.thumbnail:
                return (
                    st.thumbnail.file_id,
                    st.file_unique_id,
                    st.thumbnail.file_size or 0,
                    False,
                    ".webp",
                )
            return None, None, 0, False, ""
        return st.file_id, st.file_unique_id, st.file_size or 0, False, ".webp"

    if message.animation:
        anim = message.animation
        return anim.file_id, anim.file_unique_id, anim.file_size or 0, True, ".mp4"

    if message.photo:
        p = message.photo[-1]
        return p.file_id, p.file_unique_id, p.file_size or 0, False, ".jpg"

    if message.video:
        v = message.video
        return v.file_id, v.file_unique_id, v.file_size or 0, True, ".mp4"

    if message.video_note:
        vn = message.video_note
        return vn.file_id, vn.file_unique_id, vn.file_size or 0, True, ".mp4"

    if message.document:
        doc = message.document
        mime = (doc.mime_type or "").lower()
        if mime.startswith("image/"):
            is_gif = "gif" in mime
            return doc.file_id, doc.file_unique_id, doc.file_size or 0, is_gif, ".gif" if is_gif else ".jpg"
        if mime.startswith("video/"):
            return doc.file_id, doc.file_unique_id, doc.file_size or 0, True, ".mp4"

    return None, None, 0, False, ""


async def delete_and_notify(bot, chat_id: int, message_id: int, user_mention: str) -> None:
    """Deletes the NSFW message and sends a short formatted alert with the blue button."""
    try:
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception as e:
        logger.warning(f"Delete failed in {chat_id} for msg {message_id}: {e}")
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
        logger.warning(f"Failed to send deletion notice in {chat_id}: {e}")


async def send_size_limit_reply(bot, chat_id: int, message_id: int) -> None:
    """Replies with a short formatted notice when media exceeds 8 MB."""
    limit_text = (
        "⚠️ <b>File Limit Exceeded</b>\n\n"
        "<blockquote>This file is larger than my <b>8 MB</b> limit, so I cannot process it.</blockquote>"
    )
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=limit_text,
            reply_to_message_id=message_id,
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        logger.warning(f"Failed to send size limit notice in {chat_id}: {e}")


async def prefetch_and_enqueue(bot, job: dict) -> None:
    """
    Downloads the media file in the background (up to 3 concurrent downloads)
    and pushes the ready local file to the sequential scan queue so the scanner never waits.
    """
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
            logger.error(f"Download failed for {job['file_id']}: {e}")
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


async def handle_media_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles new and edited media messages with instant cache lookup and 8 MB check."""
    message = update.effective_message
    chat = update.effective_chat
    if not message or not chat:
        return

    file_id, unique_id, file_size, is_video_or_anim, suffix = extract_media_info(message)
    if not file_id:
        return

    if file_size > MAX_FILE_SIZE:
        await send_size_limit_reply(context.bot, chat.id, message.message_id)
        return

    user_mention = (
        message.from_user.mention_html() if message.from_user else "<b>User</b>"
    )

    # 1. Check cache first (instant action for repeated stickers/GIFs)
    cached_score = get_cached_score(unique_id)
    if cached_score is not None:
        if cached_score >= NSFW_THRESHOLD:
            await delete_and_notify(context.bot, chat.id, message.message_id, user_mention)
        return

    # 2. Pre-fetch download and queue for sequential one-by-one scanning
    job = {
        "chat_id": chat.id,
        "message_id": message.message_id,
        "user_mention": user_mention,
        "file_id": file_id,
        "unique_id": unique_id,
        "is_video_or_anim": is_video_or_anim,
        "suffix": suffix,
    }
    asyncio.create_task(prefetch_and_enqueue(context.bot, job))


# ==================== BACKGROUND WORKERS ====================
async def webhook_processor_worker(bot_app: Application) -> None:
    """Processes raw webhook updates pulled from the instant memory queue."""
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
            logger.error(f"Webhook parse error: {e}")
        finally:
            webhook_raw_queue.task_done()


async def sequential_scan_worker(bot_app: Application) -> None:
    """Scans pre-downloaded files strictly one-by-one in the dedicated single thread."""
    loop = asyncio.get_running_loop()

    while True:
        job = await scan_queue.get()
        temp_path = job.get("temp_path")
        try:
            # Double-check cache in case an identical sticker finished moments ago
            cached = get_cached_score(job["unique_id"])
            if cached is not None:
                score = cached
            else:
                score = await loop.run_in_executor(
                    scan_executor,
                    scan_media_sync,
                    temp_path,
                    job["is_video_or_anim"],
                )
                set_cached_score(job["unique_id"], score)

            if score >= NSFW_THRESHOLD:
                await delete_and_notify(
                    bot_app.bot,
                    job["chat_id"],
                    job["message_id"],
                    job["user_mention"],
                )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Scan worker error: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            scan_queue.task_done()


async def self_ping_loop() -> None:
    """Pings the server every 30 seconds to keep the Render instance awake."""
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
        logger.info(f"Webhook set to: {webhook_url}")

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
    """Immediately pushes raw bytes into memory and returns 200 OK in <1ms."""
    body = await request.body()
    if body:
        webhook_raw_queue.put_nowait(body)
    return Response(status_code=200)


if __name__ == "__main__":
    uvicorn.run("code:app", host="0.0.0.0", port=PORT, log_level="info")
