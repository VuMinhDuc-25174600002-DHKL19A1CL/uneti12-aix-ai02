from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image
from inference_bridge import discover_models, predict_pil

MODELS = {
    'mobilenet_v3_large': 'ai02_mobilenet_v3_large.pth',
    'improved': 'improved.joblib',
    'baseline': 'baseline.joblib',
}

def main() -> int:
    parser = argparse.ArgumentParser(description='AI02 CLI: predict one waste image')
    parser.add_argument('--image', required=True, help='Image path')
    parser.add_argument('--model', choices=MODELS, default='mobilenet_v3_large')
    args = parser.parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        parser.error(f'Image not found: {image_path}')
    model_path = Path('model') / MODELS[args.model]
    available = discover_models('model')
    selection = next((m for m in available if model_path in m['paths']), None)
    if selection is None:
        parser.error(f'Model file missing: {model_path}')
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    result = predict_pil(image, selection)
    print(f"Model: {selection['name']}")
    print(f"Prediction: {result.get('label_vi', result.get('label'))} ({result.get('label')})")
    print(f"Confidence (reference): {float(result.get('confidence', 0))*100:.1f}%")
    for label, value in sorted(result['probabilities'].items(), key=lambda item: item[1], reverse=True):
        print(f'  {label}: {value*100:.1f}%')
    if result.get('warning'):
        print('Warning:', result['warning'])
    if result.get('recommendation'):
        print('Advice:', result['recommendation'])
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
