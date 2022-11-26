import wget


def dowanload(in_url: str, out_folder):
    wget.download(in_url, out = out_folder)
