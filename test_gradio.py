from dotenv import load_dotenv
load_dotenv()

import os
import threading
import gradio as gr

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq, translate_text
from voice_of_the_doctor import text_to_speech_with_gtts

system_prompt = """You are a professional doctor. Based on the input, provide a structured response including:

- **Diagnosis:** Identify the possible condition based on symptoms.
- **Medications:** Recommended medical treatments (if applicable).
- **Home Remedies:** Natural or cost-free remedies for relief.
- **Precautions & Lifestyle Changes:** Preventative steps and lifestyle modifications.
- **Urgency Level:** Whether immediate medical attention is needed.

Keep your response concise (max 3-4 sentences). No preamble; start directly.
"""

def process_inputs(audio_filepath, image_filepath):
    transcription_result = []

    def transcribe():
        transcribed_text, detected_lang = transcribe_with_groq(
            stt_model="whisper-large-v3",
            audio_filepath=audio_filepath
        )
        transcription_result.append((transcribed_text, detected_lang))

    thread = threading.Thread(target=transcribe)
    thread.start()
    thread.join()

    if transcription_result:
        speech_to_text_output, detected_lang = transcription_result[0]
    else:
        speech_to_text_output, detected_lang = "Error in transcription", "en"

    # Step 2: Process the image input
    if image_filepath:
        doctor_response = analyze_image_with_query(
            query=system_prompt + " " + speech_to_text_output,
            encoded_image=encode_image(image_filepath),
            model="llama-3.2-11b-vision-preview"
        )
    else:
        doctor_response = "No image provided for analysis."

    # Step 3: Translate the doctor's response to detected language
    translated_response = translate_text(doctor_response, detected_lang)

    # Step 4: Convert the translated text to speech in the detected language
    text_to_speech_with_gtts(input_text=translated_response, output_filepath="final.mp3", language=detected_lang)

    return speech_to_text_output, translated_response, "final.mp3"

iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio("final.mp3")
    ],
    title="AI Doctor (Multi-Language Support)"
)

if __name__ == "__main__":
    iface.launch(debug=True)
