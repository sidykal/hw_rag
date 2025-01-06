import os

#os.environ["OPENAI_API_KEY"] = ''
os.environ["LANGCHAIN_TRACING_V2"]= "true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]=''
os.environ["LANGCHAIN_PROJECT"]="pr-frosty-lime-73"