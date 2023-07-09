from flask import Flask, render_template, request
import sys
import os
from llama_index import KeywordTableIndex,SimpleDirectoryReader,LLMPredictor,ServiceContext, VectorStoreIndex, ListIndex
from llama_index import load_index_from_storage, StorageContext
from langchain.chat_models import ChatOpenAI
from deepgram import Deepgram
# from pydub import AudioSegment
# from moviepy.editor import VideoFileClip
import json
import io

DEEPGRAM_API_KEY = '682f172faae69d43baece80781177391e74dcc6b'

app = Flask(__name__)

# define LLMs
#os.environ["OPENAI_API_KEY"] = ""
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
query_engine = 0
index = 0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/singlefile')
def otherpage():
    return render_template('singlefile.html')

@app.route('/submit', methods=['POST'])
def submit():
    query = request.form.get('query')
    print(f"query: '{query}'")
    #index_ = load_existing_index(service_context)
    #query_engine_ = index_.as_query_engine()
    response = query_engine.query(query)
    print(response)

    return render_template('singlefile.html', output=response.response)

@app.route('/upload', methods=['POST'])
def upload():
    adnotes = request.form.get("notes")
    directory = "recordings"
    if not os.path.exists(directory):
        os.makedirs(directory)

    file = request.files['audio_file']
    # Check if the file is empty
    if file.filename == '':
        return 'No file selected', 400

    # Check the file extension
    file_ext = os.path.splitext(file.filename)[1].lower()

    # if file_ext == '.mp4' or file_ext == '.mov':
    #     print("this is running lol")
    #     # Extract audio from MP4 or MOV file
    #     video = VideoFileClip(file)
    #     file = video.audio#file = extract_audio(file)
    #     # Save the extracted audio file
    #     #audio.save('output.mp3')
    #     # Do further processing with the extracted audio file
    #     # ...

    export_transcription(file, "audio/mp4", adnotes)

    documents = SimpleDirectoryReader(directory).load_data()
    global index
    global query_engine
    index = ListIndex.from_documents(documents)
    #index = load_existing_index(service_context)
    query_engine = index.as_query_engine()
    # if file.filename == '':
    #     return "No file selected."

    # filename = file.filename
    return render_template('singlefile.html')

def load_existing_index(service_cntx):
    absolute_path = os.path.abspath('llama_index')
    storage_context = StorageContext.from_defaults(persist_dir=absolute_path)
    index_ = load_index_from_storage(storage_context, service_context=service_cntx)
    return index_


def request_transcription(audio, mimetype_):
    # Initializes the Deepgram SDK
    dg_client = Deepgram(DEEPGRAM_API_KEY)

    source = {'buffer': audio, 'mimetype': mimetype_}
    options = {"punctuate": True, "model": "enhanced", "language": "en-US", "diarize": True}

    print('Requesting transcript...')

    response = dg_client.transcription.sync_prerecorded(source, options)
    print("got transcript")
    return json.dumps(response, indent=4), response

def export_transcription(audio, mimetype_, ad_notes):
    j, r = request_transcription(audio, mimetype_)
    #print(r)
    words = r["results"]["channels"][0]["alternatives"][0]["words"]

    curr_speaker = words[0]["speaker"]

    # Open a file in write mode
    txtfile_path = "recordings/"+audio.filename[:-3]+"txt"
    file = open(txtfile_path, "w")
    file.write(ad_notes + "\n")
    file.write("Speaker "+str(curr_speaker)+": ")

    line = ""
    for word in words:
        if word["speaker"]==curr_speaker:
            line = line + " " + word["punctuated_word"]
        else:
            curr_speaker = word["speaker"]
            file.write(line+"\n")
            line = "Speaker "+str(word["speaker"])+": " + word["punctuated_word"]
    file.close()

    return file

# def extract_audio(video_file):
#     # # Load the video file
#     # audio = AudioSegment.from_file(video_file)
#     #
#     # # Extract the audio
#     # audio = audio.set_channels(1)  # Convert stereo to mono if needed
#
#     # Create an in-memory file-like object
#     audio_buffer = io.BytesIO()
#
#     # Set the audio codec to mp3
#     audio_codec = 'mp3'
#
#     # Read the video file into a moviepy video clip
#     video_clip = VideoFileClip(video_file.stream)
#
#     # Extract audio from the video clip and write it to the buffer
#     video_clip.audio.write_audiofile(audio_buffer, codec=audio_codec)
#
#     # Set the buffer position to the beginning
#     audio_buffer.seek(0)
#
#     # You can now use the audio buffer as needed, e.g., save it to disk or send it as a response
#     # For example, if you want to send it as a response, you can return it as follows:
#     return audio_buffer.getvalue()

if __name__ == '__main__':
    app.run(debug=True)