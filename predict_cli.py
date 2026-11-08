"""Console demo without Streamlit: python predict_cli.py --image path.jpg"""
import argparse
from src.predict import predict_image

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='Đường dẫn ảnh muốn kiểm tra')
    ap.add_argument('--model', default='model/improved.joblib')
    a=ap.parse_args()
    result=predict_image(a.model, a.image)
    print('Nhóm rác:',result['label'])
    if result['confidence'] is not None:
        print('Xác suất tham khảo:',f"{result['confidence']*100:.1f}%")
        for label,p in sorted(result['probabilities'].items(),key=lambda z:z[1],reverse=True):
            print('   ',label, f'{100*p:.1f}%')
    print('Gợi ý:',result['guide'])
    print('Lưu ý:',result['caution'])
