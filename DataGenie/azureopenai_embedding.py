from openai import AzureOpenAI
import time
from vanna.base import VannaBase

class AzureOpenAI_Embeddings:
    def __init__(self, config=None):
        VannaBase.__init__(self, config=config)
        self.AzureOpenAIclient = AzureOpenAI(
        api_version= config["api_version"],
        azure_endpoint= config["api_base"],
        api_key=config["api_key"])

    def generate_embedding(self,text):
        '''
        Generate embeddings from string of text.
        This will be used to vectorize data and user input for interactions with Azure OpenAI.
        '''
        self.AzureOpenAIclient
        response =  self.AzureOpenAIclient.embeddings.create(
            input=text, model="text-embedding-ada-002")
        embeddings = response.data[0].embedding
        time.sleep(0.1) # rest period to avoid rate limiting on AOAI for free tier
        return embeddings