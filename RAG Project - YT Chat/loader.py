from langchain_community.document_loaders import YoutubeLoader

def yt_transcripter(url):
    loader = YoutubeLoader.from_youtube_url(url, language='english', add_video_info=False)
    docs = loader.load()
    return docs