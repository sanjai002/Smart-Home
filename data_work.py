import telegram
import cv2
count=5
bot = telegram.Bot(token='7162908022:AAGjz5T2hS_mvIZl3kUqocVlg3edg3rdr24')
image_path = f'img{count}.jpg'
#bot.sendMessage(chat_id='688027432', text='Hello, World!')

try:
    bot.sendPhoto(chat_id='688027432', photo=open('img1.jpg', 'rb'))
except Exception as e:
    print("Error sending photo via Telegram:", e)
                    
