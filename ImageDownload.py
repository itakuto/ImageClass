from google_images_download import google_images_download

res = google_images_download.googleimagesdownload()
arguments = {"keywords":"ばら",
             "limit": 10,
             "format": "png"
             }

res.download(arguments)
