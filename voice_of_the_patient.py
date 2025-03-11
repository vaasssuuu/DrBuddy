from dotenv import load_dotenv
load_dotenv()

import os
import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
from groq import Groq
from langdetect import detect
from deep_translator import GoogleTranslator  # Translation added

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path, timeout=20, phrase_time_limit=None):
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")
            
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            
            logging.info(f"Audio saved to {file_path}")

    except Exception as e:
        logging.error(f"Error: {e}")

def transcribe_with_groq(stt_model, audio_filepath):
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
            )
        
        transcribed_text = transcription.text
        detected_lang = detect(transcribed_text) 

        return transcribed_text, detected_lang
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        return "", "en"

def translate_text(text, target_lang):
    try:
        if target_lang == "en":
            return text
        return GoogleTranslator(source="auto", target=target_lang).tranwslate(text)
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text  
