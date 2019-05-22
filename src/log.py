from PyQt5.QtCore import pyqtSignal, QObject
import threading
import multiprocessing


# 可以考虑单例模式
class log_class(QObject):
    receiveLogSignal = pyqtSignal(str)  # log信号
    def __init__(self):
        super(log_class, self).__init__()
        self.logQueue = multiprocessing.Queue()  # 日志队列
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True) # 设置为daemon，槽函数里面为while循环
        self.logOutputThread.start()

    # 日志队列一旦有消息，就立刻发送到相应的槽里面（执行显示）
    def receiveLog(self):
        while True:
            data = self.logQueue.get()
            if data:
                self.receiveLogSignal.emit(data)
            else:
                continue
