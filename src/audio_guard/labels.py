"""Door Event 모델의 클래스와 외부 이벤트 계약입니다."""

LABELS = {"background": 0, "knock": 1, "handle": 2}
CLASS_NAMES = tuple(LABELS)
EVENTS = {
    "background": "NO_EVENT",
    "knock": "KNOCK_EVENT",
    "handle": "HANDLE_EVENT",
}
EVENT_CLASSES = ("knock", "handle")


def class_name(class_index):
    try:
        return CLASS_NAMES[int(class_index)]
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Unknown class index: {class_index}") from error
