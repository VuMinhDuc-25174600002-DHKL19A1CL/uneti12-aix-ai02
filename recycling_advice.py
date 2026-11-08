"""Explain *potential* recyclability; the image classifier does not detect composition or local rules.

This is a cautious, editable knowledge layer, NOT a second trained classifier.
Sources: https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables
         https://www.epa.gov/recycle/frequent-questions-recycling
"""
from __future__ import annotations

from typing import Mapping

LABELS = ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash')
MATERIAL_VI = dict(cardboard='Bìa carton', glass='Thủy tinh', metal='Kim loại',
                   paper='Giấy', plastic='Nhựa', trash='Rác khác')
CONDITIONS = ('Không rõ', 'Sạch / đã loại bỏ chất bẩn', 'Bẩn, dính thực phẩm / dầu mỡ')
COLLECTION = ('Chưa biết điểm thu gom có nhận không',
              'Đã xác nhận điểm thu gom nhận ĐÚNG loại này',
              'Điểm thu gom hiện tại không nhận loại này')

STEPS = {
    'cardboard': ('Gỡ băng dính/chi tiết không phải giấy nếu có; gấp phẳng thùng.',
                  'Kiểm tra lớp phủ nhựa, vết bẩn và yêu cầu thu gom địa phương.'),
    'paper': ('Giữ giấy sạch, khô; tách vật liệu phủ nhựa, nilon nếu có.',
              'Giấy phủ màng, giấy nhiễm bẩn hoặc loại đặc biệt có thể không được thu gom.'),
    'metal': ('Đổ hết thức ăn/chất lỏng, làm sạch lon; phân loại nếu địa phương yêu cầu.',
              'Không cho bình áp suất còn chất bên trong, pin hoặc vật nguy hại vào rác tái chế thông thường.'),
    'glass': ('Kiểm tra đó có phải chai/lọ thủy tinh được điểm thu gom chấp nhận.',
              'Không bỏ mảnh thủy tinh sắc nhọn hoặc bóng đèn vào thùng tái chế thường; xử lý theo nơi thu gom.'),
    'plastic': ('Xem ký hiệu/chủng loại nhựa và hỏi nơi thu gom có nhận loại này không.',
               'Tháo bỏ thức ăn; nhiều loại túi màng, xốp và nhựa phức hợp không được thu gom phổ thông.'),
    'trash': ('Quan sát vật thật để xác định thành phần trước khi quyết định.',
              'Nhóm “rác khác” gồm nhiều loại vật phẩm, không đồng nghĩa “không thể tái chế”.'),
}


def assess_recycling(label: str, *, condition: str = CONDITIONS[0],
                     collection: str = COLLECTION[0], special: bool = False,
                     prediction_uncertain: bool = False) -> dict:
    """Rules of thumb, not a yes/no derived from the image alone.

    status: yes/conditional/no_for_collection/separate/unknown.
    'yes' means conditional on user-confirmed condition and local acceptance.
    """
    if label not in LABELS:
        label = 'trash'
    if condition not in CONDITIONS:
        condition = CONDITIONS[0]
    if collection not in COLLECTION:
        collection = COLLECTION[0]
    steps = STEPS[label]
    base = {
        'material': MATERIAL_VI[label], 'steps': list(steps),
        'conditions': condition, 'local_collection': collection,
        'source': 'https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables',
        'warning': 'Chỉ là hướng dẫn theo nhóm vật liệu; ảnh không xác nhận được loại nhựa, lớp phủ, độ sạch hay quy định nơi thu gom.',
    }
    if special:
        return dict(base, status='separate', title='CẦN THU GOM RIÊNG', answer='Không bỏ vào thùng tái chế thông thường',
                    explanation='Bạn đã đánh dấu vật phẩm đặc biệt/nguy hại: pin, điện tử, bóng đèn, hóa chất, vật sắc nhọn… Hãy hỏi điểm thu gom chuyên biệt.')
    if prediction_uncertain:
        return dict(base, status='unknown', title='CHƯA ĐỦ CĂN CỨ', answer='Chưa thể kết luận',
                    explanation='Mô hình chưa chắc chắn về vật liệu. Hãy kiểm tra vật thể bằng mắt hoặc chụp thêm góc trước khi quyết định tái chế.')
    if label == 'trash':
        return dict(base, status='unknown', title='CHƯA XÁC ĐỊNH', answer='Chưa thể kết luận',
                    explanation='“Rác khác” không cho biết vật liệu cụ thể; cần xem xét vật thật để xác định nơi xử lý.')
    if collection == COLLECTION[2]:
        return dict(base, status='no_for_collection', title='KHÔNG THU GOM TẠI ĐIỂM NÀY',
                    answer='Không bỏ vào thùng tái chế tại đây',
                    explanation='Bạn xác nhận điểm thu gom này không nhận đúng loại vật liệu/vật phẩm đó. Tìm điểm thu gom phù hợp khác.')
    if condition == CONDITIONS[2]:
        return dict(base, status='conditional', title='CHƯA NÊN BỎ VÀO THÙNG TÁI CHẾ',
                    answer='Cần làm sạch hoặc kiểm tra trước',
                    explanation='Vật phẩm đang bẩn/dính thực phẩm hoặc dầu mỡ. Nếu có thể làm sạch theo yêu cầu nơi thu gom, hãy xử lý trước.')
    if condition == CONDITIONS[1] and collection == COLLECTION[1]:
        return dict(base, status='yes', title='CÓ THỂ TÁI CHẾ THEO THÔNG TIN BẠN CUNG CẤP',
                    answer='Có – nếu vật phẩm thực sự thuộc loại được chấp nhận',
                    explanation='Bạn đã xác nhận tình trạng sạch và điểm thu gom nhận đúng loại này. Vẫn cần kiểm tra trực tiếp đặc tính sản phẩm.')
    if label == 'plastic':
        return dict(base, status='conditional', title='CÓ TIỀM NĂNG TÁI CHẾ, CẦN XÁC MINH',
                    answer='Có điều kiện – còn phụ thuộc loại nhựa',
                    explanation='Nhận diện “nhựa” không xác định được mã nhựa hoặc loại bao bì; cần hỏi nơi thu gom nhận đúng loại.')
    return dict(base, status='conditional', title='CÓ TIỀM NĂNG TÁI CHẾ',
                answer='Có thể – cần xác minh tình trạng và nơi thu gom',
                explanation='Nhóm vật liệu này thường có thể được tái chế, nhưng còn phụ thuộc loại sản phẩm, độ sạch và địa phương.')
