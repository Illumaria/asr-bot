import os

import requests
from telegram import Update
from telegram.ext import (
    CallbackContext,
    CommandHandler,
    Filters,
    MessageHandler,
    Updater,
)

API_TOKEN = os.getenv("API_TOKEN")
BACKEND_URL = os.getenv("BACKEND_URL")
URL = f"{BACKEND_URL}/predict"


def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued"""
    update.message.reply_text("Ready to transcribe your audio messages! :3")


def send_transcribe_request(update: Update, context: CallbackContext) -> None:
    """Send a request to the backend to transcribe the audio message"""
    update.message.reply_text("Got your voice message! Transcribing...")

    voice_message_file_id = update.message.voice.file_id
    voice_message_file = context.bot.get_file(voice_message_file_id)
    voice_message_byte_array = voice_message_file.download_as_bytearray()
    response = requests.post(
        url=URL,
        data=voice_message_byte_array,
        headers={"Content-Type": "application/octet-stream"},
    )
    update.message.reply_text(f"You said: {response.text}")


def main() -> None:
    """Start the bot"""
    if not API_TOKEN:
        raise KeyError("API_TOKEN not found in environment variables")
    if not BACKEND_URL:
        raise KeyError("BACKEND_URL not found in environment variables")

    updater = Updater(API_TOKEN)

    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start, run_async=True))
    dispatcher.add_handler(
        MessageHandler(
            Filters.voice & ~Filters.command, send_transcribe_request, run_async=True
        )
    )

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
