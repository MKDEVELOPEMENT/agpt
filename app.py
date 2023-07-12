from flask import Flask, render_template, request
import os
from llama_index import KeywordTableIndex,SimpleDirectoryReader,LLMPredictor,ServiceContext, VectorStoreIndex, ListIndex
from llama_index import load_index_from_storage, StorageContext
from langchain.chat_models import ChatOpenAI
from deepgram import Deepgram
import json
import shutil
from google.cloud import storage

DEEPGRAM_API_KEY = '682f172faae69d43baece80781177391e74dcc6b'
os.environ["GCLOUD_PROJECT"] = "agpt-389322"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'agpt-389322-a2f03394a344.json'

app = Flask(__name__)

# define LLMs
#os.environ["OPENAI_API_KEY"] = ""
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
#query_engine = 0
index = 0

@app.route('/')
def home():
    indices = get_unique_indices()
    return render_template('singlefile.html', dropdown_options=indices)

@app.route('/singlefile')
def otherpage():
    return render_template('singlefile.html')

@app.route('/submit', methods=['POST'])
def submit():
    query = request.form.get('query')
    print(f"query: '{query}'")
    index_ = load_existing_index(service_context)
    query_engine = index_.as_query_engine()
    # query_engine_serialized = session.get('query_engine')
    # if query_engine_serialized is None:
    #     return 'Query engine not found', 400
    #
    # query_engine = dill.loads(query_engine_serialized)
    response = query_engine.query(query)
    print(response)

    return render_template('singlefile.html', output=response.response)

@app.route('/upload', methods=['POST'])
def upload():
    subfolder = "temp_index"
    if os.path.exists(subfolder):
        # Delete all contents inside the subfolder
        for root, dirs, files in os.walk(subfolder):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))
    else:
        os.makedirs(subfolder)

    selected_option = request.form.get('dropdown_menu')

    if selected_option:
        download_index_files_gcs(selected_option)
        return render_template("singlefile.html")

    adnotes = request.form.get("notes")
    directory = "recordings"
    if os.path.exists(directory):
        # Delete all contents inside the subfolder
        for root, dirs, files in os.walk(directory):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))
    else:
        os.makedirs(directory)

    file = request.files['audio_file']
    # Check if the file is empty
    if file.filename == '':
        return 'No file selected', 400

    # Check the file extension
    file_pre = os.path.splitext(file.filename)[0]

    export_transcription(file, "audio/mp4", adnotes)

    documents = SimpleDirectoryReader(directory).load_data()
    index = ListIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="temp_index")

    #save index to google cloud
    upload_blob("temp_index/docstore.json", "indices/"+file_pre+"/docstore.json")
    upload_blob("temp_index/graph_store.json", "indices/" + file_pre + "/graph_store.json")
    upload_blob("temp_index/index_store.json", "indices/" + file_pre + "/index_store.json")
    upload_blob("temp_index/vector_store.json", "indices/" + file_pre + "/vector_store.json")

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

def upload_blob(source_file_name, destination_blob_name):
    bucket_name = "agpt_bucket1"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    storage_client = storage.Client()
    bloblist = []

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    # Note: The call returns a response only when the iterator is consumed.
    print("Blobs:")
    for blob in blobs:
        print(blob.name)
        bloblist.append(blob.name)

    return bloblist

def get_unique_indices():
    bloblist = list_blobs_with_prefix("agpt_bucket1", "indices/")
    index_list = [blob.split('/')[1] for blob in bloblist]
    uniques = list(set(index_list))
    return uniques

def download_blob(source_blob_name, destination_file_name):
    bucket_name = "agpt_bucket1"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )

def download_index_files_gcs(indx):
    download_blob("indices/"+indx+"/docstore.json", "temp_index/docstore.json")
    download_blob("indices/" + indx + "/graph_store.json", "temp_index/graph_store.json")
    download_blob("indices/" + indx + "/index_store.json", "temp_index/index_store.json")
    download_blob("indices/" + indx + "/vector_store.json", "temp_index/vector_store.json")

if __name__ == '__main__':
    app.run(debug=True)