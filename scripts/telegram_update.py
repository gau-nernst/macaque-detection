import telebot
import requests

API_KEY = "ENTER BOT TOKEN"
CHANNEL_ID = "CH_ID"
DEVICE_NAME = "LOCATION"

bot = telebot.TeleBot(API_KEY)

@bot.message_handler(commands=["Help"])
def help_Message(message):
    out_message= "Hi there, I am a Maccaque Detection bot"
    bot.send_message(message.chat.id, out_message)

def call_Debug(message):
    return message.text.lower() == 'debug'

@bot.message_handler(func=call_Debug)
def debug_Message(message):
    out_message = f"Maccaque spotted at {DEVICE_NAME}, do be careful."
    # Following command used to send message directly to channel
    requests.post(f'https://api.telegram.org/bot{API_KEY}/sendMessage?chat_id={CHANNEL_ID}&text={out_message}')

    bot.send_message(message.chat.id, out_message)

bot.polling()