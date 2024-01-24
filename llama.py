import os
import lamini
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("LAMINI_API_KEY")

lamini.api_key = os.environ.get("LAMINI_API_KEY")

llm = lamini.LlamaV2Runner()
print(llm("How are you?"))
