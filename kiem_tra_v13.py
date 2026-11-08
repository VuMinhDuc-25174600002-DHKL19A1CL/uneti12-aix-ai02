"""V13 diagnostic: checks files and packages without ever loading untrusted model."""
from pathlib import Path
from importlib.util import find_spec
from sys import version
ROOT=Path(__file__).resolve().parent

def main():
    print('='*65)
    print('AI02 V13 - KIEM TRA DA GOC + TAI CHE')
    print('Project:',ROOT,'\nPython:',version.split()[0])
    print('='*65)
    for module in ('streamlit','PIL','numpy','sklearn','torch','torchvision'):
        try: installed=find_spec(module) is not None
        except Exception: installed=False
        print('[OK]' if installed else '[THIEU]',module)
    for fname in ('app.py','recycling_advice.py','multiview_inference.py',
                  'inference_bridge.py','cnn_utils.py','ui_v10_utils.py','src/predict.py'):
        print('[OK]' if (ROOT/fname).exists() else '[THIEU]',fname)
    models=sorted((ROOT/'model').glob('*.pth'))+sorted((ROOT/'model').glob('*.joblib'))
    print('Checkpoint that:', [p.name for p in models] if models else 'CHUA CO')
    print('Mo: http://localhost:8501')
    print('Anh da goc: 2-6 ANH THAT CUA CUNG MOT VAT, khac huong chup.')
    print('Tai che: thong tin tham khao, KHONG phai chung nhan thu gom.')

if __name__ == '__main__': main()
