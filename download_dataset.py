"""Download original TrashNet resized ZIP from the source repository (Internet required).

No dataset is re-distributed in this submitted solution package.
"""
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
import shutil

URL='https://github.com/garythung/trashnet/raw/refs/heads/master/data/dataset-resized.zip'
TARGET=Path('data')

def main():
    TARGET.mkdir(exist_ok=True)
    zip_path=TARGET/'dataset-resized.zip'
    print('Đang tải TrashNet ~41 MB từ kho gốc...', URL)
    try:
        with urlopen(Request(URL,headers={'User-Agent':'Mozilla/5.0'}),timeout=90) as resp, zip_path.open('wb') as out:
            shutil.copyfileobj(resp,out)
        with ZipFile(zip_path) as zf:
            members=zf.namelist()
            for entry in zf.infolist():
                dest=(TARGET / entry.filename).resolve()
                if not dest.is_relative_to(TARGET.resolve()):
                    raise ValueError('Không giải nén vì ZIP có đường dẫn vượt thư mục đích')
            zf.extractall(TARGET)
        candidates=list(TARGET.rglob('dataset-resized'))
        print('Đã giải nén. Thư mục dữ liệu có thể là:',[str(x) for x in candidates if x.is_dir()])
    except Exception as ex:
        print('TẢI TỰ ĐỘNG KHÔNG THÀNH CÔNG:',ex)
        print('Tải thủ công theo nguồn: https://github.com/garythung/trashnet')
        print('Giải nén vào data/dataset-resized gồm 6 thư mục con.')
        raise

if __name__=='__main__':main()
