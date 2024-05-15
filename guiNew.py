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


# class MplCanvas(FigureCanvas):
#     def __init__(self, parent=None, width=15, height=4, dpi=100, channels=4):
#         fig = Figure(figsize=(width, height), dpi=dpi, facecolor='black')
#         gs = fig.add_gridspec(channels, hspace=0)
#         self.axes = gs.subplots()
#         super(MplCanvas, self).__init__(fig)
#         # fig.tight_layout()
          

class GUI(QtWidgets.QMainWindow):
    def __init__(self, model):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = uic.loadUi('GUI\GUInew.ui',self)
        
        self.model = model

        self.pgwin = pg.PlotWidget()
        

        self.threadpool = QtCore.QThreadPool()	
        self.q = queue.Queue(maxsize=20)

        self.window_length = 1000
        self.downsample = 1
        self.n_channels = 10
        self.interval = 4 
        self.samplerate = 250
        self.length = 500

        self.file = "Session1_Sub4_Class2_MI_100LP_Resampled250.csv"
        self.eeg = pd.read_csv(self.file, usecols=range(2,32)).to_numpy()

        # self.canvas = MplCanvas(self, width=15, height=5, dpi=100, channels=self.n_channels)
        self.ui.gridLayout_2.addWidget(self.pgwin, 0, 1, 1, 1)
        self.reference_plots = []
        
        self.plotdata =  np.zeros((self.length,self.n_channels))
        self.update_plot()
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.interval) #msec
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

        self.startBtn.clicked.connect(self.start_worker)
        self.stopBtn.clicked.connect(self.stop_stream)
        self.stopBtn.setEnabled(False)
        self.running = True
        
        
    def dataStream(self):
        try:
            i=0
            while self.running:
                 self.q.put(self.eeg[i,0:self.n_channels]/1000)
                 self.model.q.put(self.eeg[i,0:30])
                 time.sleep(1/250)
                 i+=1
        except Exception as e:
            print("ERROR: ",e)

    def classify(self):
        try:
            while self.running:
                output = self.model.run()
                binary_output = (output >= 0.5).int()
                self.accLbl.setText(str(binary_output))
        except Exception as e:
            print("ERRORdada: ",e)

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
                # self.canvas.axes.set_facecolor((0,0,0)) 
                self.max = 10
                self.min = -10
                
                if len(self.reference_plots)==0:
                    self.pgwin.setYRange(-15, 12*self.n_channels)
                    for i in range(self.n_channels):
                        self.reference_plots.append(self.pgwin.plot(self.ydata[:,i]+12*i, pen=(i,3)))
                    # for ax in range(self.n_channels):
                    #     plot_refs = self.canvas.axes[ax].plot(self.ydata[:,ax], color=(0,1,0.12), linewidth=0.5)
                    #     self.reference_plots.append(plot_refs[0])
                    #     self.canvas.axes[ax].axis('off')	
                        
                else:
                    for i in range(self.n_channels):
                        self.reference_plots[i].setData(self.ydata[:,i]+12*i)
                    # for plot in range(self.n_channels):
                    #     self.reference_plots[plot].set_ydata(self.ydata[:,plot])

            # for plot in range(self.n_channels):
            #     self.canvas.axes[plot].set_ylim( ymin=self.min, ymax=self.max)		
            # self.canvas.draw()
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