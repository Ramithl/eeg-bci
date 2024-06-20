import sys
# import matplotlib
# matplotlib.use('Qt5Agg')
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
import queue
import numpy as np
import pandas as pd
import pdb
from PyQt5 import QtCore, QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
import time
from queue import Queue
import pyqtgraph as pg


class GUI(QtWidgets.QMainWindow):
    def __init__(self, model):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = uic.loadUi('GUI\GUInew.ui',self)
        
        self.model = model

        self.pgwin = pg.PlotWidget()
        

        self.threadpool = QtCore.QThreadPool()	
        self.q = queue.Queue(maxsize=20)

        self.accuracy = None
        self.total_predictions = 0
        self.correct_predictions = 0

        self.window_length = 1000
        self.downsample = 1
        self.n_channels = 10
        self.samplerate = 100
        self.interval = 10
        self.length = 500
        self.trial_size = 500

        self.file = "sub1ses1_cls4_MI_100LP_Resampled_5.csv"
        self.eeg1 = pd.read_csv(self.file, usecols=range(1,32)).to_numpy()

        reshaped_arr = self.eeg1.reshape(-1, self.trial_size, self.eeg1.shape[1])
        np.random.shuffle(reshaped_arr)
        self.eeg = reshaped_arr.reshape(self.eeg1.shape)

        self.ui.gridLayout_2.addWidget(self.pgwin, 0, 1, 1, 1)
        self.reference_plots = []
        
        self.plotdata =  np.zeros((self.length,self.n_channels))
        self.update_plot()
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.interval) #msec
        self.timer.timeout.connect(self.update_plot)
        # self.timer.start()

        self.startBtn.clicked.connect(self.start_worker)
        self.stopBtn.clicked.connect(self.stop_stream)
        self.stopBtn.setEnabled(False)
        self.running = True

        # self.lineEdit.textChanged['QString'].connect(self.update_window_length)

        self.accValue.setText("N/A")
        
        
    def dataStream(self):
        try:
            i=0
            while self.running:
                 self.q.put(self.eeg[i,1:self.n_channels+1]/1000) # +1 to include class
                 self.model.q.put(self.eeg[i,0:31])
                 time.sleep(1/250)
                 i+=1
        except Exception as e:
            print("STREAM ERROR: ",e)

    def classify(self):
        try:
            while self.running:
                output, true_label = self.model.run()
                if true_label == None:
                    continue
                predicted_label = np.argmax(output.detach().numpy().flatten())
                self.update_labels(true_label, predicted_label)
                
                self.total_predictions += 1
                if true_label == predicted_label:
                        self.correct_predictions += 1

                self.accuracy = (self.correct_predictions/self.total_predictions)*100
                
                self.accValue.setText(f"{self.accuracy:.2f}%")
        except Exception as e:
            print("MODEL ERROR: ",e)

    def start_worker(self):
        self.running = True
        self.model.running = True
        self.startBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        self.timer.start()
        worker = Worker(self.start_stream, )
        classifier = Worker(self.classify, )
        self.threadpool.start(worker)
        self.threadpool.start(classifier)	

    def start_stream(self):
        self.dataStream()
    
    def stop_stream(self):
        self.startBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)
        self.running = False
        self.model.stop()
        self.timer.stop()

    def update_labels(self, true_label, predicted_label):

        self.true_0.setStyleSheet("background-color: rgb(245, 245, 245); border: 2px solid; border-color: rgb(185, 185, 185);")
        self.true_1.setStyleSheet("background-color: rgb(245, 245, 245); border: 2px solid; border-color: rgb(185, 185, 185);")
        self.true_2.setStyleSheet("background-color: rgb(245, 245, 245); border: 2px solid; border-color: rgb(185, 185, 185);")
        self.true_3.setStyleSheet("background-color: rgb(245, 245, 245); border: 2px solid; border-color: rgb(185, 185, 185);")

        self.predicted_0.setStyleSheet("background-color: rgb(245, 245, 245); border: 2px solid; border-color: rgb(185, 185, 185);")
        self.predicted_1.setStyleSheet("background-color: rgb(245, 245, 245); border: 2px solid; border-color: rgb(185, 185, 185);")
        self.predicted_2.setStyleSheet("background-color: rgb(245, 245, 245); border: 2px solid; border-color: rgb(185, 185, 185);")
        self.predicted_3.setStyleSheet("background-color: rgb(245, 245, 245); border: 2px solid; border-color: rgb(185, 185, 185);") 

        style = "background-color: rgb(221, 255, 213); border: 2px solid; border-color: rgb(0, 209, 0);"
        trueLbl = getattr(self, f'true_{int(true_label)}')  
        trueLbl.setStyleSheet(style)     

        if true_label == predicted_label:
            style_predicted = "background-color: rgb(221, 255, 213); border: 2px solid; border-color: rgb(0, 209, 0);"
            predictedLbl = getattr(self, f'predicted_{int(true_label)}')
        
        else:
            style_predicted = "background-color: rgb(255, 206, 206); border: 2px solid red;"
            predictedLbl = getattr(self, f'predicted_{int(predicted_label)}')

        predictedLbl.setStyleSheet(style_predicted)
            
            
        
    # def update_window_length(self,value):
    #     self.window_length = int(value)
    #     length  = 50
    #     self.plotdata =  np.zeros((length,len(self.channels)))
    #     self.update_plot()

    def update_plot(self):
        try:
            self.errorLbl.setText("")
            self.errorLbl.setStyleSheet("background-color: rgb(245, 245, 245); border: 1px solid; border-color: rgb(185, 185, 185);")
            data=[0]
            while True:
                try: 
                    data = self.q.get_nowait()
                except queue.Empty:
                    break
                shift = 1
                self.plotdata = np.roll(self.plotdata, -shift, axis = 0)
                self.plotdata[-shift:,:] = data
                self.ydata = self.plotdata[:]
                self.max = 10
                self.min = -10
                
                if len(self.reference_plots)==0:
                    self.pgwin.setYRange(-15, 12*self.n_channels)
                    for i in range(self.n_channels):
                        self.reference_plots.append(self.pgwin.plot(self.ydata[:,i]+12*i, pen=(1,3)))
                        
                else:
                    for i in range(self.n_channels):
                        self.reference_plots[i].setData(self.ydata[:,i]+12*i)

        except Exception as e:
            self.errorLbl.setText(str(e))
            self.errorLbl.setStyleSheet("background-color: rgb(255, 226, 226); color: black; border: 1px solid red;")

class Worker(QtCore.QRunnable):

    def __init__(self, function, *args, **kwargs):
        super(Worker, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        self.function(*self.args, **self.kwargs)		


# app = QtWidgets.QApplication(sys.argv)
# mainWindow = GUI()
# mainWindow.show()
# sys.exit(app.exec_())