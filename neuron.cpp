#include "neuron.h"

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(QObject *parent, unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}

void Neuron::setOutputVal(double val)
{
    m_outputVal = val;
}

double Neuron::getOutputVal() const
{
    return m_outputVal;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    for (int n = 0; n < prevLayer.size(); ++n)
    {
        sum += prevLayer[n]->getOutputVal() *
                prevLayer[n]->m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    for (int n = 0; n < prevLayer.size(); ++n)
    {
        Neuron *&neuron = prevLayer[n];
        double oldDeltaWeight = neuron->m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
                eta
                * neuron->getOutputVal()
                * m_gradient
                + alpha
                * oldDeltaWeight;

        neuron->m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron->m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

ConnectionList Neuron::getConnectionList()
{
    return m_outputWeights;
}

double Neuron::randomWeight()
{
    return qrand() / double(RAND_MAX);
}

double Neuron::transferFunction(double x)
{
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    return 1.0 - x * x;
}

double Neuron::sumDOW(const Layer &nextLayer)
{
    double sum = 0.0;

    for (int n = 0; n < nextLayer.count() - 1; ++n)
    {
        sum += m_outputWeights[n].weight * nextLayer[n]->m_gradient;
    }
    return sum;
}
