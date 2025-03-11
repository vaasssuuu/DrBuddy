from dotenv import load_dotenv
load_dotenv()

import os
import platform
import subprocess
from gtts import gTTS

def text_to_speech_with_gtts(input_text, output_filepath, language="en"):
    try:
        audioobj = gTTS(text=input_text, lang=language, slow=False)
        audioobj.save(output_filepath)
    except ValueError:
        print(f"Language '{language}' is not supported by gTTS. Defaulting to English.")
        audioobj = gTTS(text=input_text, lang="en", slow=False)
        audioobj.save(output_filepath)

    # Play the audio
    os_name = platform.system()
    try:
        if os_name == "Darwin": 
            subprocess.run(['afplay', output_filepath])
        elif os_name == "Windows": 
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'])
        elif os_name == "Linux": 
            subprocess.run(['aplay', output_filepath])  
        else:
            raise OSError("Unsupported OS")
    except Exception as e:
        print(f"Error playing the audio: {e}")
