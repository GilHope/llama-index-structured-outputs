import os
from dotenv import load_dotenv

import nest_asyncio

nest_asyncio.apply()

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Load OpenAI API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key is missing!")

llm = OpenAI(model="gpt-4o")
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = llm
Settings.embed_model = embed_model

from typing import List
from pydantic import BaseModel, Field


class Song(BaseModel):
    """Data model for a song."""

    title: str
    length_seconds: int


class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str
    songs: List[Song]

from llama_index.core.llms import ChatMessage

sllm = llm.as_structured_llm(output_cls=Album)
input_msg = ChatMessage.from_str("Generate an example album from The Shining")

### SYNC ###
# output = sllm.chat([input_msg])
# # get actual object
# output_obj = output.raw

# print(str(output))
# print(output_obj)

### ASYNC ###
import asyncio

# async def main():
#     output = await sllm.achat([input_msg])
#     # get actual object
#     output_obj = output.raw
#     print(str(output))
#     return output_obj
# asyncio.run(main())

### SYNC STREAMING ###
from IPython.display import clear_output
from pprint import pprint

stream_output = sllm.stream_chat([input_msg])
for partial_output in stream_output:
    clear_output(wait=True)
    pprint(partial_output.raw.dict())

output_obj = partial_output.raw
print(str(output))

### ASYNC STREAMING ###
# from IPython.display import clear_output

# stream_output = await sllm.astream_chat([input_msg])
# async for partial_output in stream_output:
#     clear_output(wait=True)
#     pprint(partial_output.raw.dict())