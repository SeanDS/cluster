import sys
from PyQt5 import Qt, QtGui, QtCore, QtWidgets

from .geometric import DistanceConstraint

def solution_viewer(problem, solution):
    window = Example()

    for component in solution:
        # coordinates
        position = solution[component]

        window.draw_circle(position, component)

    for constraint in problem.constraint_graph.constraints():
        if not isinstance(constraint, DistanceConstraint):
            continue

        (componentA, componentB) = constraint.variables()

        window.draw_line(solution[componentA], solution[componentB])

    window.show()

class Example():
    def __init__(self):
        self.initUI()

    def initUI(self):
        # create application
        self.qApplication = Qt.QApplication([])
        self.qMainWindow = Qt.QMainWindow()
        self.qMainWindow.setGeometry(150, 150, 700, 700) # x, y, w, h

        # set close behaviour to prevent zombie processes
        self.qMainWindow.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        # create drawing area
        self.qScene = QtWidgets.QGraphicsScene()

        # create view
        self.qView = QtWidgets.QGraphicsView(self.qScene, self.qMainWindow)
        self.qView.setSceneRect(self.qScene.sceneRect())
        self.qView.setFixedSize(700, 700)

        # set window title
        self.qMainWindow.setWindowTitle('Optivis')

    def draw_circle(self, position, text=None):
        circle = QtWidgets.QGraphicsEllipseItem(position.x - 5, position.y - 5, 10, 10)

        if text is not None:
            circle.setToolTip(str(text))

        self.qScene.addItem(circle)

    def draw_line(self, pointA, pointB):
        line = QtWidgets.QGraphicsLineItem(pointA.x, pointA.y, pointB.x, pointB.y)

        self.qScene.addItem(line)

    def show(self):
        self.qMainWindow.show()

        sys.exit(self.qApplication.exec_())

def main():
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
