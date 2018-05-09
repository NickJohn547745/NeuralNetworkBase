#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QFile dataFile("C:/Users/Nicholas/Desktop/data.txt");
    qDebug() << dataFile.open(QIODevice::ReadOnly | QIODevice::Text);
    QTextStream in(&dataFile);

    QVector<double> inputVals,
            targetVals,
            resultVals;

    QVector<unsigned> topology;

    QString topologyLine = in.readLine();
    topologyLine.replace("topology: ", "");
    foreach (QString value, topologyLine.split(' '))
    {
        topology << value.toInt();
        ui->label_4->setText(ui->label_4->text() + " - " + QString::number(value.toInt()));
    }

    myNet = new Net(this, topology);

    int count = 0;

    while (!in.atEnd())
    {
        count++;
        inputVals.clear();
        targetVals.clear();

        QString line = in.readLine();

        if (line.contains("in:"))
        {
            line.replace("in: ", "");
            foreach (QString value, line.split(' '))
            {
                inputVals << value.toDouble();
            }

            myNet->feedForward(inputVals);
            myNet->getResults(resultVals);

            ui->listWidgetInput->addItem(line.replace(" ", "    -    "));
            foreach (double result, resultVals)
            {
                ui->listWidgetOutput->addItem(QString::number(result, 'f', 1));
            }
        }
        else if (line.contains("out:"))
        {
            line.replace("out: ", "");
            foreach (QString value, line.split(' '))
            {
                targetVals << value.toDouble();
            }
            ui->listWidgetCorrect->addItem(line);
            myNet->backProp(targetVals);
        }
        qDebug() << "Net recent average error: " << myNet->getRecentAverageError() << "\n";
    }
    qDebug() << dataFile.size();
    qDebug() << "Sucessfully processed " << count << " lines!";

    myNet->getWeights();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButtonTest_clicked()
{
    QVector<double> inputVals,
            targetVals,
            resultVals;

    ui->listWidgetInput->addItem(QString::number(ui->doubleSpinBoxValueOne->value()) + "    -    " + QString::number(ui->doubleSpinBoxValueTwo->value()));

    inputVals << ui->doubleSpinBoxValueOne->value() << ui->doubleSpinBoxValueTwo->value();
    myNet->feedForward(inputVals);

    myNet->getResults(resultVals);

    foreach (double result, resultVals)
    {
        ui->listWidgetOutput->addItem(QString::number(result, 'f', 1));
    }

    targetVals << ui->doubleSpinBoxCorrect->value();

    ui->listWidgetCorrect->addItem(QString::number(ui->doubleSpinBoxCorrect->value()));

    myNet->backProp(targetVals);


}
