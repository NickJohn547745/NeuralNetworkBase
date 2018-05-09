#ifndef NET_H
#define NET_H

#include <QObject>
#include <QDebug>

#include "neuron.h"

typedef QVector<Neuron *> Layer;

class Net : public QObject
{
    Q_OBJECT

public:
    explicit Net(QObject *parent, const QVector<unsigned> &topology);
    void feedForward(const QVector<double> &inputVals);
    void backProp(const QVector<double> &targetVals);
    void getResults(QVector<double> &resultVals) const;
    double getRecentAverageError() const;
    QVector<QVector<double> > getWeights();

signals:

public slots:

private:
    QVector<Layer> m_layers;
    double m_error;
    double m_recentAverageError;
    double m_recentAverageErrorSmoothingFactor;
};

#endif // NET_H
