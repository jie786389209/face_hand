from PyQt5.QtCore import pyqtSignal, QObject
import threading
import multiprocessing


# ���Կ��ǵ���ģʽ
class log_class(QObject):
    receiveLogSignal = pyqtSignal(str)  # log�ź�
    def __init__(self):
        super(log_class, self).__init__()
        self.logQueue = multiprocessing.Queue()  # ��־����
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True) # ����Ϊdaemon���ۺ�������Ϊwhileѭ��
        self.logOutputThread.start()

    # ��־����һ������Ϣ�������̷��͵���Ӧ�Ĳ����棨ִ����ʾ��
    def receiveLog(self):
        while True:
            data = self.logQueue.get()
            if data:
                self.receiveLogSignal.emit(data)
            else:
                continue
