#ifndef NEURON_H
#define NEURON_H

#include <QObject>
#include <QDebug>

class Neuron;

typedef QVector<Neuron *> Layer;

struct Connection
{
    double weight;
    double deltaWeight;
};

struct ConnectionList : public QVector<Connection>
{
public:
    QString toString()
    {
        QString final = "";
        for (int x = 0; x < this->count(); x++)
        {
            final += "(";
            final += QString::number(at(x).weight);
            final += ", ";
            final += QString::number(at(x).deltaWeight);
            final += ")";
        }
        return final;
    }
    QString weightToString()
    {
        QString final = "";
        for (int x = 0; x < this->count(); x++)
        {
            final += "(";
            final += QString::number(at(x).weight);
            final += ")";
        }
        return final;
    }
    QString weightStringAt(int index)
    {
        return QString::number(at(index).weight);
    }
    QString deltaStringAt(int index)
    {
        return QString::number(at(index).deltaWeight);
    }
};

class Neuron : public QObject
{
    Q_OBJECT

public:
    explicit Neuron(QObject *parent, unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val);
    double getOutputVal() const;
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer  &nextLayer);
    void updateInputWeights(Layer &prevLayer);
    ConnectionList getConnectionList();

private:
    static double eta;
    static double alpha;
    static double randomWeight(void);
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    double sumDOW(const Layer &nextLayer);
    double m_outputVal;
    ConnectionList m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

#endif // NEURON_H
