import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from recognise import recognise
from download import get_size
from PIL import Image
from train import progress, train, test, model, downloadMNIST
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import NavigationToolbar2
import time
import sys
from torch import nn, optim, cuda
import torch
from torch.utils import data
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from matplotlib.axes._subplots import Axes
from matplotlib.figure import Figure
import random
# plt settings
plt.rcdefaults()
# path definition
sys.path.append(os.path.realpath('..'))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# global variable
prob = [0] * 10
count = 0
modelpath = './models/conv.pth'

transform = transforms.Compose([transforms.Grayscale(
    num_output_channels=1), lambda x: transforms.functional.invert(
    x), transforms.ToTensor()])
trainset = datasets.MNIST(
    r'mnist_data/', transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=60000)
dataiter = iter(trainloader)  # creating a iterator
# creating images for image and lables for image number (0 to 9)
images, labels = dataiter.next()

# main window


class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
        global prob

    def initUI(self):
        self.setWindowTitle('Team 21')
        self.resize(660, 350)
        self.centre()
        self.menu()
        self.viewCanvas()
        self.show()

    def centre(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def menu(self):
        menuBar = self.menuBar()
        # Required for macos, there is a bug, remove this and menu bar will not show up on macos
        menuBar.setNativeMenuBar(False)

        # Exit Action with a shortcut
        exitAction = QAction(
            QIcon('scripts/icons/exit.png'), '    Exit    ', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        # Train Action with a shortcut
        trainAction = QAction(
            QIcon(''), '    Train    ', self)
        trainAction.setShortcut('Ctrl+R')
        trainAction.setStatusTip('Train a Model')
        trainAction.triggered.connect(self.trainModelPopup_t)

        # Add file menu items
        fileMenu = menuBar.addMenu('&File')
        fileMenu.addAction(trainAction)
        fileMenu.addAction(exitAction)

        # View Training Image Action
        viewTrainAction = QAction(
            QIcon(''), ' View Training Image  ', self)
        viewTrainAction.setStatusTip('View Training Image')
        viewTrainAction.triggered.connect(self.viewTrain_t)

        # View Testing Image Action
        viewTestAction = QAction(
            QIcon(''), ' View Testing Image  ', self)
        viewTestAction.setStatusTip('View Testing Image')
        viewTestAction.triggered.connect(self.viewTest_t)

        # Add view menu items
        viewMenu = menuBar.addMenu('&View')
        viewMenu.addAction(viewTrainAction)
        viewMenu.addAction(viewTestAction)

    # when called, display the training images with pop-up window
    def viewTrain_t(self):
        self.viewWindow = viewTrain()

    # when called, display the testing images with pop-up window
    def viewTest_t(self):
        self.viewWindow = viewTest()

    # handler for window pop-up
    def trainModelPopup_t(self):
        self.window = trainModelWindow()
        self.window.show()

    # buttons and labels initialization on the main screen
    def viewCanvas(self):
        # Define canvas
        drawer = drawingWindow()
        probability = Probability()

        # Define buttons
        # Clear Button
        clearbtn = QPushButton(
            'Clear')
        clearbtn.clicked.connect(
            self.clearprob)
        clearbtn.clicked.connect(
            probability.update_plot)
        clearbtn.clicked.connect(
            lambda: [drawer.clearImage(), drawer.saveImage("scripts/images/1/0.jpg", "JPEG")])

        # Random Button
        randombtn = QPushButton('Random', self)
        randombtn.clicked.connect(
            self.randomModel)

        # Select a Model Button
        modelbtn = QPushButton('Model', self)
        modelbtn.clicked.connect(
            self.selectmodel)

        # Recognise Button
        self.recognizebtn = QPushButton('Recognize')
        self.recognizebtn.clicked.connect(
            lambda: drawer.saveImage("scripts/images/1/0.jpg", "JPEG"))
        self.recognizebtn.clicked.connect(
            self.clickprocess)
        self.recognizebtn.clicked.connect(
            probability.update_plot)
        self.recognizebtn.clicked.connect(
            self.findMax)

        # Define frames
        main = QFrame()
        main.setFrameShape(QFrame.Box)

        # Model Button
        self.modelSelected = QLabel()
        self.modelSelected.setText("conv.pth")
        self.modelSelected.setFont(QFont('Arial', 20))
        self.modelSelected.setAlignment(Qt.AlignCenter)

        # Result Label
        self.result = QLabel()
        self.result.setAlignment(Qt.AlignCenter)
        self.result.setText("")
        self.result.setFont(QFont('Arial', 30))

        # Introduce splitters
        splitter1 = QSplitter(Qt.Horizontal)
        splitter2 = QSplitter(Qt.Vertical)
        splitter3 = QSplitter(Qt.Vertical)

        # The 3 side splits:
        # Buttons
        splitter2.addWidget(clearbtn)
        splitter2.addWidget(randombtn)
        splitter2.addWidget(modelbtn)
        splitter2.addWidget(self.recognizebtn)
        # The other 2
        splitter2.addWidget(probability)
        splitter2.addWidget(self.result)

        # main drawing section
        splitter1.addWidget(drawer)
        splitter1.addWidget(splitter2)
        splitter1.setStretchFactor(0, 0)

        splitter3.addWidget(self.modelSelected)
        splitter3.addWidget(splitter1)

        # defining layouts
        hbox = QHBoxLayout()
        widget = QWidget()
        hbox.addWidget(splitter3)
        self.setCentralWidget(widget)
        widget.setLayout(hbox)
        self.setWindowTitle('Team 02')
        self.show()

    # Use a model to identify the drawn image
    def clickprocess(self):
        global prob, modelpath
        temp_prob = [0] * 10
        for i in range(1):
            plt.close()
            prob = recognise(modelpath, 'scripts/images/')
            temp = prob.index(max(prob))
            temp_prob[temp] = temp_prob[temp] + 1

        best = temp_prob.index(max(temp_prob))
        while (prob.index(max(prob)) != best):
            prob = recognise(modelpath, 'scripts/images/')

    # finds a digit with highest probability
    def findMax(self):
        global prob
        self.result.setText(str(prob.index(max(prob))))

    # Clear the probability chart
    def clearprob(self):
        global prob
        prob = [0] * 10

    # File dialog for selecting a model from a folder
    def selectmodel(self):
        global modelpath
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            './models', "Model files (*.pth)")
        modelpath = fname[0]
        self.modelSelected.setText(os.path.basename(modelpath))

    # Randomly choose a model from a folder
    def randomModel(self):
        global modelpath
        dirModels = os.listdir('./models')
        n = random.randint(1, len(dirModels))
        modelpath = './models/' + dirModels[n-1]
        self.modelSelected.setText(os.path.basename(dirModels[n-1]))

# Training Images View


class viewTrain(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Training Dataset')
        self.plot()

    # Use QScroll to display an array of Qpixmap
    def plot(self):

        # define the scroll area
        self.scrollArea = QScrollArea(widgetResizable=True)
        self.scrollArea.setBackgroundRole(QPalette.Dark)

        content_widget = QWidget()
        self.scrollArea.setWidget(content_widget)
        lay = QVBoxLayout(content_widget)
        lay.setSpacing(0)

        # iterate all 600 images containing 100 digits each
        for i in range(1, 600):
            path = 'dataset/' + str(i) + '.jpg'
            qImg = QImage(path)
            self.imageLabel = QLabel()
            self.imageLabel.setBackgroundRole(QPalette.Base)
            self.imageLabel.setSizePolicy(QSizePolicy.Ignored,
                                          QSizePolicy.Ignored)  # not sure about this one.
            self.imageLabel.setPixmap(QPixmap(qImg).scaled(
                400, 300, QtCore.Qt.KeepAspectRatio))
            self.imageLabel.adjustSize()
            self.imageLabel.setFixedSize(self.imageLabel.size())
            lay.addWidget(self.imageLabel)

        # include it in the window
        lay.setContentsMargins(0, 0, 0, 0)
        box = QHBoxLayout()
        box.addWidget(self.scrollArea)
        self.setLayout(box)
        self.show()

# View Testing images


class viewTest(QWidget):
    def __init__(self):
        # Initialization of window
        super().__init__()
        self.setWindowTitle('Test Dataset')
        self.resize(500, 500)
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Testing Dataset')
        self.image = QImage(300, 300, QImage.Format_Grayscale8)
        self.plot()

    def new_home(self, *args, **kwargs):
        pass

    # Hook matplotlib's forward button
    def new_next(self, *args, **kwargs):

        global count, images, labels
        if (count <= 60000):
            plt.close()
            count = count + 1
            data = images.reshape(len(images), 28, 28, 1)
            image = np.asarray(data[count]).squeeze()
            plt.title(str(labels[count].numpy()), fontsize=30)
            plt.imshow(image, cmap="gray")
            plt.show()

    # Hook matplotlib's backward button
    def new_back(self, *args, **kwargs):

        global count, images, labels
        if (count > 0):
            plt.close()

            count = count - 1
            data = images.reshape(len(images), 28, 28, 1)
            image = np.asarray(data[count]).squeeze()
            plt.imshow(image, cmap="gray")
            plt.title(str(labels[count].numpy()), fontsize=30)
            plt.show()

    #  Plot the image
    def plot(self):
        plt.close()
        self.home = NavigationToolbar2.home
        fig = plt.figure()
        fig.canvas.set_window_title('Testing Dataset')
        NavigationToolbar2.home = self.new_home
        NavigationToolbar2.forward = self.new_next
        NavigationToolbar2.back = self.new_back

        self.i = 0
        global count, images, labels
        data = images.reshape(len(images), 28, 28, 1)
        image = np.asarray(data[count]).squeeze()
        plt.imshow(image, cmap="gray")
        plt.title(str(labels[count].numpy()), fontsize=30)
        plt.show()

# Probability graph view


class Probability(FigureCanvas):
    def __init__(self, parent=None):
        global prob
        self.fig = Figure()
        self.axes = self.fig.add_subplot(111)

        # Add 10 bars
        self.Product = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        # import probability from the global variable
        self.axes.barh(self.Product, prob)
        self.axes.axes.xaxis.set_visible(False)
        yvalues = np.arange(10)
        self.axes.set_yticklabels(yvalues, fontsize=3)
        super(Probability, self).__init__(self.fig)

    # for updating the plot with new probability
    # INPUT: Current displayed graph axes
    # OUTPUT: NA
    def update_plot(self, axes):
        global prob
        # link displayed graph for updating its value
        self.axes = self.fig.gca()

        # clear the axes
        self.axes.clear()
        yvalues = np.arange(10)
        self.axes.set_yticklabels(yvalues, fontsize=3)

        # import new data and draw
        self.axes.barh(self.Product, prob)
        self.draw()

# drawing canvas window


class drawingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Team 21')

        self.setFixedSize(300, 300)
        self.image = QImage(300, 300, QImage.Format_Grayscale8)
        self.image.fill(Qt.white)
        self.drawing = False

    # when left mouse button is clicked
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True  # drawing started
            self.lastPoint = event.pos()  # draw

    # updates mouse's location and draw a line inbetween the last position
    def mouseMoveEvent(self, event):
        if(event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, 40, Qt.SolidLine))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    # when left mouse button is released
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False  # stop drawing

    # save the drawn image for recognition
    # INPUT: File name and format for saving the image
    # OUTPUT: softmaxed distribution array
    def saveImage(self, fileName, fileFormat):
        self.image.save(fileName, fileFormat)

    # clear the canvas
    def clearImage(self):
        self.path = QPainterPath()
        self.image.fill(Qt.white)  # switch it to else
        self.update()

    # QPainter initialization
    def paintEvent(self, event):
        canvas = QPainter(self)
        canvas.drawImage(self.rect(), self.image, self.image.rect())


# popup window for Train Model
class trainModelWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Train')

        # enable custom window hint
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.CustomizeWindowHint)

        # disable (but not hide) close button
        self.setWindowFlags(self.windowFlags() & ~
                            QtCore.Qt.WindowCloseButtonHint)

        self.trainModelPopup()

    # initializa training window
    def trainModelPopup(self):
        global progress
        grid = QGridLayout()

        # download button
        downloadBtn = QPushButton('Download MNIST', self)
        downloadBtn.clicked.connect(
            self.download_start)

        # train button
        trainBtn = QPushButton('Train', self)
        trainBtn.clicked.connect(
            self.train_start)

        # cancel/exit button
        cancelBtn = QPushButton('Cancel', self)
        cancelBtn.clicked.connect(self.close)

        self.dialogue = QTextEdit('')
        self.pbar = QProgressBar(self)

        self.setWindowTitle('Dialogue')
        grid.addWidget(self.dialogue, 1, 1, 4, 15)
        grid.addWidget(self.pbar, 5, 1, 1, 20)
        grid.addWidget(downloadBtn, 7, 2, 1, 2)
        grid.addWidget(trainBtn, 7, 6, 1, 2)
        grid.addWidget(cancelBtn, 7, 10, 1, 2)
        self.setLayout(grid)
        self.show()

        # initialise all QThread
        # update download progress in background thread
        self.download_thread = DownloadThread()
        self.download_thread._signal.connect(self.signal_accept)
        self.download_thread._end_signal.connect(
            self.signal_epoch_string_accept)

        # download the dataset in the background
        self.download_MNIST_thread = DownloadMNISThread()
        self.download_MNIST_thread._signal.connect(self.signal_accept)

        # training a model in the background
        self.train_thread = TrainThread()
        self.train_thread._signal.connect(self.signal_accept)
        self.train_thread._epoch_string_signal.connect(
            self.signal_epoch_string_accept)

    # Training start
    def train_start(self):
        # make sure the entire mnist data exists
        if (get_size('mnist_data') < 100.0):
            self.signal_epoch_string_accept(
                "MNIST dataset was not available.   Please download...")
        elif (get_size('mnist_data') >= 100.0):
            self.signal_accept(0)
            downloadMNIST()
            self.signal_epoch_string_accept(
                "Training has started... epoch=75")
            self.setEnabled(False)
            self.train_thread.start()
        else:
            self.signal_epoch_string_accept(
                "Error Training")

    # Downloading start
    def download_start(self):
        # initialize
        self.signal_accept(0)
        self.setEnabled(False)
        self.signal_epoch_string_accept(
            "Downloading...")
        self.download_thread.start()
        self.download_MNIST_thread.start()

    # progress bar initialization
    # waiting for signal containing the progress value
    def signal_accept(self, msg):
        self.pbar.setValue(int(msg))
        if self.pbar.value() >= 99:
            self.pbar.setValue(100)
            self.setEnabled(True)

    # dialogue text initialization
    def signal_epoch_string_accept(self, msg):
        self.dialogue.append(msg)
        self.dialogue.moveCursor(QtGui.QTextCursor.End)

    # make sure QThreads are closed and terminated when the training window is force closed
    # prevent crash and overflow
    def closeEvent(self, event):
        if (self.train_thread.isRunning):
            self.train_thread.stop()
            self.train_thread.wait()
        elif (self.download_thread.isRunning):
            self.download_thread.stop()
            self.download_thread.wait()

# Training


class TrainThread(QThread):
    # Initialize signals for progress bar, output diaplogue and finished signal
    _signal = pyqtSignal(int)
    _epoch_string_signal = pyqtSignal(str)
    _epoch_exit_signal = pyqtSignal(bool)

    def __init__(self):
        super(TrainThread, self).__init__()
        self.exit = False

    # Start
    def run(self):
        global progress
        max = 75  # defiend epoch
        since = time.time()
        # iterate n=max times
        for epoch in range(1, max):
            if (self.exit == True):
                break
            self._epoch_string_signal.emit(
                'Training Epoch : {}'.format(epoch))
            progress += 100/(max-1)
            epoch_start = time.time()
            train(epoch)
            m, s = divmod(time.time() - epoch_start, 60)
            self._epoch_string_signal.emit(
                f'Training time: {m:.0f}m {s:.0f}s')
            self._epoch_string_signal.emit(test())
            m, s = divmod(time.time() - epoch_start, 60)

            time.sleep(0.1)
            self._signal.emit(progress)
        if (self.exit == False):
            torch.save(model.state_dict(), './models/train.pth')

    # stop training when called from external sources
    def stop(self):
        self.exit = True

# Download Progress


class DownloadThread(QThread):
    # Initialize signals for progress bar and finished signal
    _signal = pyqtSignal(int)
    _end_signal = pyqtSignal(str)

    def __init__(self):
        super(DownloadThread, self).__init__()

    # size of mnist_data is calculated for the progress bar
    def run(self):
        self._signal.emit(get_size('mnist_data'))
        # updates the progress bar every time the size changes
        while (get_size('mnist_data') < 100.0):
            time.sleep(1)
            self._signal.emit(get_size('mnist_data'))
        self._end_signal.emit("MNIST downloaded successfully")

    # stop training when called from external sources
    def stop(self):
        self.terminate()
        self.wait()


# Download MNIST
class DownloadMNISThread(QThread):
    # Initialize signal for progress bar
    _signal = pyqtSignal(int)

    def __init__(self):
        super(DownloadMNISThread, self).__init__()

    def run(self):
        # Downloads in the background
        downloadMNIST()

    # called to terminate the QThread and stops downloading
    def stop(self):
        self.terminate()
        self.wait()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
