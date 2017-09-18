from appJar import gui
from EigenFaces import webcamController as main


def press(button):
    if button == "Training":
        try:
            main.mainAll(path=app.getEntry("Path Training"), size=app.getEntry("size input"),
                         output=app.getEntry("Path Training"))
        except Exception as e:
            print("Error: ", str(e))
            main.mainAll()
            raise
    elif button == "Test":
        try:
            main.testload(path=app.getEntry("Path Testing"), size=app.getEntry("size testing"))
        except Exception as e:
            print("Error: ", str(e))
            main.testload()
            raise
    elif button == "Start Webcam":
        try:
            main.mainWebcam(nama_file=app.getEntry("*.NPZ File"), channel=app.getEntry("channel"))
        except Exception as e:
            print("Error: ", str(e))
            main.mainWebcam(channel=1)
            raise


def getDir(button):
    # lok = app.directoryBox().dirName
    if button == "Browse\nroot path":
        lok = app.directoryBox()
        app.setEntry("Path Training", lok)
    elif button == "Browse \nnpz Directory":
        lok = app.directoryBox()
        app.setEntry("Path Testing", lok)
    elif button == "Browse \nNPZ File":
        fil = app.openBox(fileTypes=[("Numpy File", "*.npz")])
        app.setEntry("*.NPZ File", fil)


app = gui("Face Recognition using eigenfaces", "640x480")
app.startTabbedFrame("TabbedFrame")
app.setTabbedFrameTabExpand("TabbedFrame")

# Training
app.startTab("Training")
app.startLabelFrame("Training Details")
app.setFont(18)
app.addLabelEntry("Path Training", 0, 0)
app.addButton("Browse\nroot path", getDir, 0, 1)
app.addLabelEntry("size input")
app.setFocus("Path Training")
app.addButton("Training", press, 2, 0)
app.stopLabelFrame()
app.stopTab()

# Testing
app.startTab("Evaluasi")
app.startLabelFrame("Evaluasi Details")
app.setFont(18)
app.addLabelEntry("Path Testing", 0, 0)
app.addButton("Browse \nnpz Directory", getDir, 0, 1)
app.addLabelEntry("size testing")
app.setFocus("Path Testing")
app.addButton("Test", press, 2, 0)
app.stopLabelFrame()
app.stopTab()

# Webcam
app.startTab("Webcam")
app.startLabelFrame("Webcam Details")
app.setFont(18)
app.addLabelEntry("*.NPZ File", 0, 0)
app.addButton("Browse \nNPZ File", getDir, 0, 1)
app.addLabelEntry("size")
app.addLabelEntry("channel")
app.setFocus("*.NPZ File")
app.addButton("Start Webcam", press)
app.stopLabelFrame()
app.stopTab()

app.stopTabbedFrame()

app.go()
