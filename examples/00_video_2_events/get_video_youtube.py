# From https://dev.to/seijind/how-to-download-youtube-videos-in-python-44od
import pytube
url = 'https://www.youtube.com/watch?v=RtUQ_pz5wlo'
youtube = pytube.YouTube(url)
video = youtube.streams.get_highest_resolution()
video.download("../../data/video/")