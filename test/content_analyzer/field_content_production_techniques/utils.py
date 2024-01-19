import json
import os

import requests

from clayrs.content_analyzer import JSONFile
from test import dir_test_files

videos_path = os.path.join(dir_test_files, 'test_videos', 'videos_files')
audios_path = os.path.join(dir_test_files, 'test_audios', 'audios_files')
images_path = os.path.join(dir_test_files, 'test_images', 'images_files')


def dump_ds(to_write: list, key: str, ds_path):

    local_paths_ds = []
    for i, val in enumerate(to_write):
        local_paths_ds.append({key: val, "itemID": i})

    with open(ds_path, "w") as f:
        json.dump(local_paths_ds, f)


def get_multimedia_file(url: str, files_path, local_filename=None):

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:

        filename = local_filename
        if filename is None:
            filename = os.path.basename(url)
            if '?' in filename:
                filename = filename.split('?')[0]

        filepath = os.path.join(files_path, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)
    else:
        raise ValueError("Couldn't download test file")

    return filename


def create_video_dataset():
    if not os.path.isdir(videos_path):
        os.makedirs(videos_path)

        videos_list = ["https://raw.githubusercontent.com/pytorch/vision/main/test/assets/videos/WUzgd7C1pWA.mp4",
                       "https://raw.githubusercontent.com/pytorch/vision/main/test/assets/videos/SOX5yA1l24A.mp4",
                       "https://raw.githubusercontent.com/pytorch/vision/main/test/assets/videos/R6llTwEh07w.mp4"]

        local_source_path = os.path.join(videos_path, "..", "paths_dataset.json")
        url_source_path = os.path.join(videos_path, "..", "urls_dataset.json")

        local_paths = []
        video_path_base = os.path.join("test", "test_files", "test_videos", "videos_files")

        for video in videos_list:
            filename = get_multimedia_file(video, videos_path)
            local_paths.append(os.path.join(video_path_base, filename))

        dump_ds(local_paths, "videoPath", local_source_path)
        dump_ds(videos_list, "videoUrl", url_source_path)


def create_audio_dataset():
    if not os.path.isdir(audios_path):
        os.makedirs(audios_path)

        audios_list = [
            "https://github.com/librosa/librosa-test-data/raw/72bd79e448829187f6336818b3f6bdc2c2ae8f5a/test2_8000.wav",
            "https://github.com/pytorch/audio/raw/main/test/torchaudio_unittest/assets/vad-go-mono-32000.wav",
            "https://github.com/pytorch/audio/raw/main/test/torchaudio_unittest/assets/kaldi_file_8000.wav",
        ]

        local_source_path = os.path.join(audios_path, "..", "paths_dataset.json")
        url_source_path = os.path.join(audios_path, "..", "urls_dataset.json")

        local_paths = []
        audio_path_base = os.path.join("test", "test_files", "test_audios", "audios_files")

        for audio in audios_list:
            filename = get_multimedia_file(audio, audios_path)
            local_paths.append(os.path.join(audio_path_base, filename))

        dump_ds(local_paths, "audioPath", local_source_path)
        dump_ds(audios_list, "audioUrl", url_source_path)


def create_image_dataset():
    if not os.path.isdir(images_path):
        os.makedirs(images_path)

        images_list = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Tombeau_Valentine_Balbiani_details_Pilon_LP397_MR1643.png/1200px-Tombeau_Valentine_Balbiani_details_Pilon_LP397_MR1643.png?20120424204328",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/M.Nonius_Balbus_Nuceria_equestrian_Statue_MANN.png/909px-M.Nonius_Balbus_Nuceria_equestrian_Statue_MANN.png?20161017094632",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Aurige._Mus%C3%A9e_Delphes._Gr%C3%A8ce.png/622px-Aurige._Mus%C3%A9e_Delphes._Gr%C3%A8ce.png?20211012163557"
        ]

        local_source_path = os.path.join(images_path, "..", "paths_dataset.json")
        url_source_path = os.path.join(images_path, "..", "urls_dataset.json")

        local_paths = []
        image_path_base = os.path.join("test", "test_files", "test_images", "images_files")

        for image in images_list:
            filename = get_multimedia_file(image, images_path)
            local_paths.append(os.path.join(image_path_base, filename))

        dump_ds(local_paths, "imagePath", local_source_path)
        dump_ds(images_list, "imageUrl", url_source_path)


def create_full_path_source(raw_source_path_local, field_name, path) -> str:
    source = JSONFile(raw_source_path_local)
    full_source_path = os.path.join(dir_test_files, os.path.join(path, '..', 'tmp_full_source_path.json'))

    data_with_full_paths = []
    for data in source:
        data_with_full_path = data
        item_name = os.path.basename(data_with_full_path[field_name])
        data_with_full_path[field_name] = os.path.join(path, item_name)
        data_with_full_paths.append(data_with_full_path)

    with open(full_source_path, 'w') as f:
        json.dump(data_with_full_paths, f)

    return full_source_path


class MockedGoodResponse:

    def __init__(self, file_dir):
        self.file_dir = file_dir

    @property
    def content(self):
        file_path = os.path.join(self.file_dir, os.listdir(self.file_dir)[0])
        with open(file_path, "rb") as file:
            content = file.read()
        return content


class MockedUnidentifiedResponse:

    def __init__(self):
        self.content = str.encode('unidentified')
