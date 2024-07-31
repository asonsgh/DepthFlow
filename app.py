import os
import io
import requests
from PIL import Image
import random
import time
import uuid
from moviepy.editor import AudioFileClip, concatenate_audioclips, concatenate_videoclips, ImageClip
import cv2
import numpy as np
from DepthFlow import DepthScene
from ShaderFlow.Message import ShaderMessage
import json
from pprint import pprint
from g4f.client import Client


def fetch_imagedescription_and_script(prompt):
    client = Client()

    # Make the request to G4F's API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    # Extract data from the API's response
    output = json.loads(response.choices[0].message.content.strip())
    pprint(output)
    image_prompts = [k['image_description'] for k in output]
    texts = [k['text'] for k in output]

    return image_prompts, texts

# Daily motivation, personal growth and positivity
topic = "Success and Achievement"
goal = "inspire people to overcome challenges, achieve success and celebrate their victories"

prompt_prefix = """You are tasked with creating a script for a {} video that is about 30 seconds.
Your goal is to {}.
Please follow these instructions to create an engaging and impactful video:
1. Begin by setting the scene and capturing the viewer's attention with a captivating visual.
2. Each scene cut should occur every 5-10 seconds, ensuring a smooth flow and transition throughout the video.
3. For each scene cut, provide a detailed description of the stock image being shown. ONLY ENGLISH
4. Along with each image description, include a corresponding text that complements and enhances the visual. The text should be concise and powerful.
5. Ensure that the sequence of images and text builds excitement and encourages viewers to take action.
6. Strictly output your response in a JSON list format, adhering to the following sample structure:""".format(topic, goal)

sample_output = """
   [
       { "image_description": "Description of the first image here.", "text": "Text accompanying the first scene cut." },
       { "image_description": "Description of the second image here.", "text": "Text accompanying the second scene cut." },
       ...
   ]"""

prompt_postinstruction = """By following these instructions, you will create an impactful {} short-form video.
Output:""".format(topic)

prompt = prompt_prefix + sample_output + prompt_postinstruction

image_prompts, texts = fetch_imagedescription_and_script(prompt)
print("image_prompts: ", image_prompts)
print("texts: ", texts)
print(len(texts))

# Generate a unique folder name
current_uuid = uuid.uuid4()
current_foldername = str(current_uuid)
print(current_foldername)

def generate_images(prompts, fname):
    url = "https://api.segmind.com/v1/sdxl1.0-txt2img"
    headers = {'x-api-key': segmind_apikey}

    if not os.path.exists(fname):
        os.makedirs(fname)

    num_images = len(prompts)
    currentseed = random.randint(1, 1000000)
    print("seed ", currentseed)

    start_time = time.time()

    for i, prompt in enumerate(prompts):
        if i > 0 and i % 5 == 0:
            elapsed_time = time.time() - start_time
            remaining_time = 60 - elapsed_time
            if remaining_time > 0:
                print(f"Waiting for {remaining_time:.2f} seconds to comply with API rate limits...")
                time.sleep(remaining_time)
            start_time = time.time()

        final_prompt = "((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope".format(prompt.strip('.'))
        data = {
            "prompt": final_prompt,
            "negative_prompt": "((deformed)), ((limbs cut off)), ((quotes)), ((extra fingers)), ((deformed hands)), extra limbs, disfigured, blurry, bad anatomy, absent limbs, blurred, watermark, disproportionate, grainy, signature, cut off, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, amputation, disconnected limbs",
            "style": "hdr",
            "samples": 1,
            "scheduler": "UniPC",
            "num_inference_steps": 30,
            "guidance_scale": 8,
            "strength": 1,
            "seed": currentseed,
            "img_width": 1024,
            "img_height": 1024,
            "refiner": "yes",
            "base64": False
        }

        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200 and response.headers.get('content-type') == 'image/jpeg':
            image_data = response.content
            image = Image.open(io.BytesIO(image_data))

            image_filename = os.path.join(fname, f"{i + 1}.jpg")
            image.save(image_filename)

            print(f"Image {i + 1}/{num_images} saved as '{image_filename}'")
        else:
            print(response.text)
            print(f"Error: Failed to retrieve or save image {i + 1}")

def generate_and_save_audio(text, foldername, filename, voice_id, elevenlabs_apikey, model_id="eleven_multilingual_v2", stability=0.4, similarity_boost=0.80):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": elevenlabs_apikey
    }

    data = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost
        }
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code != 200:
        print(response.text)
    else:
        file_path = f"{foldername}/{filename}.mp3"
        with open(file_path, 'wb') as f:
            f.write(response.content)

def add_parallax_effect(image_path, output_path, duration):
    import moderngl
    from moderngl_window import WindowConfig
    from moderngl_window.context.osmesa import OSMesaContext

    class ParallaxEffect(WindowConfig):
        gl_version = (3, 3)
        title = "Parallax Effect"
        window_size = (1024, 1024)
        aspect_ratio = 1.0
        resizable = False

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.scene = DepthScene()
            self.scene.input(image=image_path)
            self.scene.main(output=output_path, fps=30, duration=duration)

        def render(self, time, frame_time):
            self.scene.update()

    OSMesaContext.activate()
    ParallaxEffect.run()

def create_video_clips(image_folder, audio_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, filename in enumerate(os.listdir(image_folder)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            audio_path = os.path.join(audio_folder, f"{i + 1}.mp3")
            output_path = os.path.join(output_folder, f"{i + 1}.mp4")

            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration

            parallax_image_path = os.path.join(image_folder, f"parallax_{i + 1}.mp4")
            add_parallax_effect(image_path, parallax_image_path, duration)

            image_clip = ImageClip(parallax_image_path).set_duration(duration)
            video_clip = image_clip.set_audio(audio_clip)
            video_clip.write_videofile(output_path, codec="libx264")

def create_combined_video(mp4_folder, output_filename, output_resolution=(1080, 1920), fps=24):
    mp4_files = sorted([file for file in os.listdir(mp4_folder) if file.endswith(".mp4")])
    mp4_files = sorted(mp4_files, key=lambda x: int(x.split('.')[0]))

    video_clips = [VideoFileClip(os.path.join(mp4_folder, mp4_file)) for mp4_file in mp4_files]
    final_video = concatenate_videoclips(video_clips, method="compose")
    finalpath = os.path.join(mp4_folder, output_filename)

    final_video.write_videofile(finalpath, fps=fps, codec='libx264', audio_codec="aac")

generate_images(image_prompts, current_foldername)

for i, text in enumerate(texts):
    output_filename = str(i + 1)
    print(output_filename)
    generate_and_save_audio(text, current_foldername, output_filename, voice_id, elevenlabsapi)

create_video_clips(current_foldername, current_foldername, current_foldername)

output_filename = "combined_video.mp4"
create_combined_video(current_foldername, output_filename)
