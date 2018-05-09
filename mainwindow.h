#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "net.h"

#include <QMainWindow>
#include <QDebug>
#include <QFile>
#include <QTextStream>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_pushButtonTest_clicked();

private:
    Ui::MainWindow *ui;
    Net *myNet;
};

#endif // MAINWINDOW_H
