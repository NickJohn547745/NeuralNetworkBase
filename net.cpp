#include "net.h"

Net::Net(QObject *parent, const QVector<unsigned> &topology)
    : QObject(parent)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
        {
            m_layers.back().push_back(new Neuron(this, numOutputs, neuronNum));
            qDebug() << "Made a Neuron!\n";
        }

        m_layers.back().back()->setOutputVal(1.0);
    }

}

void Net::feedForward(const QVector<double> &inputVals)
{
    Q_ASSERT(inputVals.size() == m_layers[0].size() - 1);

    for (int i = 0; i < inputVals.size(); ++i)
    {
        m_layers[0][i]->setOutputVal(inputVals[i]);
    }

    for (int layerNum = 1; layerNum < m_layers.size(); ++layerNum)
    {
        Layer prevLayer = m_layers[layerNum - 1];
        for (int n = 0; n < m_layers[layerNum].size() - 1; ++n)
        {
            m_layers[layerNum][n]->feedForward(prevLayer);
        }
    }
}

void Net::backProp(const QVector<double> &targetVals)
{
    Layer outputLayer = m_layers.back();
    m_error = 0.0;

    for (int n = 0; n < outputLayer.size() - 1; ++n)
    {
        double delta = targetVals[n] - outputLayer[n]->getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);

    m_recentAverageError =
            (m_recentAverageError * m_recentAverageErrorSmoothingFactor + m_error)
            / (m_recentAverageErrorSmoothingFactor + 1.0);

    for (int n = 0; n < outputLayer.size() - 1; ++n)
    {
        outputLayer[n]->calcOutputGradients(targetVals[n]);
    }

    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (int n = 0; n < hiddenLayer.size(); ++n)
        {
            hiddenLayer[n]->calcHiddenGradients(nextLayer);
        }
    }

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
    {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (int n = 0; n < layer.size() - 1; ++n)
        {
            layer[n]->updateInputWeights(prevLayer);
        }
    }
}

void Net::getResults(QVector<double> &resultVals) const
{
    resultVals.clear();

    for (int n = 0; n < m_layers.back().size() - 1; ++n)
    {
        resultVals.push_back(m_layers.back()[n]->getOutputVal());
    }
}

double Net::getRecentAverageError() const
{
    return m_recentAverageError;
}

QVector<QVector<double>> Net::getWeights()
{
    for (int x = 0; x < m_layers.count(); x++)
    {
        qDebug() << "Layer #" << x;
        QVector<Neuron*> neurons = m_layers[x];
        QVector<double> weights;

        for (int i = 0; i < neurons.count(); i++)
        {
            qDebug() << "Neuron #" << i;
            Neuron *neuron = neurons[i];

            qDebug() << neuron->getConnectionList().weightToString();
        }
    }
    return QVector<QVector<double>>();
}
