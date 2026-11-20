"""
MVVM 연결 도구
    Observable : 값을 담고, 값이 바뀌면 구독한 함수(View)를 호출한다
    Event      : 값 없이 알림만 보낸다 (예: 메시지 한 줄)
ViewModel 은 Observable/Event 로 상태를 내보내고, View 는 구독해서 화면만 갱신한다.
모두 GUI 스레드에서 쓴다 (백그라운드 스레드는 ViewModel 내부 상태만 바꾸고, tick() 에서 반영)
"""


class Observable:
    def __init__(self, value=None):
        self._value = value
        self._subs = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self.set(v)

    def set(self, v, force=False):
        if force or v != self._value:
            self._value = v
            for fn in list(self._subs):
                fn(v)

    def subscribe(self, fn, call_now=True):
        self._subs.append(fn)
        if call_now:
            fn(self._value)
        return fn


class Event:
    def __init__(self):
        self._subs = []

    def subscribe(self, fn):
        self._subs.append(fn)
        return fn

    def emit(self, *args):
        for fn in list(self._subs):
            fn(*args)
