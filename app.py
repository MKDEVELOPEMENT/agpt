from flask import Flask, render_template, request
import sys
import os
from llama_index import KeywordTableIndex,SimpleDirectoryReader,LLMPredictor,ServiceContext
from llama_index import load_index_from_storage, StorageContext
from langchain.chat_models import ChatOpenAI

#os.environ["OPENAI_API_KEY"] = "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"
#sys.path.append('../../AGPT')

app = Flask(__name__)

# define LLMs
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    query = request.form.get('query')
    output = "hello"  # Replace with your own logic to process the query
    index = load_existing_index(service_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(query)

    return render_template('index.html', output=response.response)

def load_existing_index(service_cntx):
    absolute_path = os.path.abspath('llama_index')
    storage_context = StorageContext.from_defaults(persist_dir=absolute_path)
    index = load_index_from_storage(storage_context, service_context=service_cntx)
    return index

if __name__ == '__main__':
    app.run(debug=True)