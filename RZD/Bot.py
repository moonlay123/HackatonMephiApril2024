import telebot
import requests
import cv2
import time
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np
from cig import pred

TOKEN = '7164817294:AAELaB8ZzFi1XivCFt1WsuCZ6XIM1mB-SW0'
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(content_types=['photo'])
def handle_photo(message):

    from pathlib import Path
    Path(f'files/{message.chat.id}/').mkdir(parents=True, exist_ok=True)
    if message.content_type == 'photo':
        file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        src = f'files/{message.chat.id}/' + file_info.file_path.replace('photos/', '')
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)

    a,b,c = pred(src)
    
    print(a,b,c)

    bot.send_message(message.chat.id, "Композиция моделей - " + str(a) + '\n' + " Ключевые точки - " + str(b) + '\n' + " Человек + сигарета - " + str(c) )

    '''
    if a and b or a and c or b and c:
        bot.send_message(message.chat.id, "Обнаружено курение!")
        #print(verd)
    else:
        bot.send_message(message.chat.id, "Курение не обнаружено!" )
        #print(verd)
    '''
        
        

bot.polling(none_stop=True)
