#include "activateFunc.h"
#include <cmath>
#include <algorithm>

double actfnc::ActivateFunc::sigmoid(double summ)
{
	return 1. / (1. + exp(-summ));
}

double actfnc::ActivateFunc::divSigmoid(double summ)
{
	return sigmoid(summ) * (1. - sigmoid(summ));
}

double actfnc::ActivateFunc::revSigmoid(double summ)
{
	return std::log(summ / (1. - summ));
}

double actfnc::ActivateFunc::relu(double summ)
{
	if (summ < 0.) return 0.01 * summ;
	else if (summ > 1.) return 1. + 0.01 * (summ - 1.);
	else return summ;
}

double actfnc::ActivateFunc::divRelu(double summ)
{
	if (summ < 0.) return 0.01;
	else if (summ > 1.) return 0.01;
	else return 1.;
}

double actfnc::ActivateFunc::revRelu(double summ)
{
	if (summ < 0.) return summ / 0.01;
	else if (summ > 1.) return 1. + (summ - 1.) / 0.01;
	else return summ;
}

actfnc::ActivateFunc::ActivateFunc() : currentType(actfnc::Sigmoid)
{}

actfnc::ActivateFunc::ActivateFunc(size_t type) :
	currentType(type)
{}

void actfnc::ActivateFunc::changeType(size_t newType)
{
	currentType = newType;
}

double actfnc::ActivateFunc::actFunc(double summ)
{
	switch (currentType)
	{
	case Relu: return relu(summ);
	case Sigmoid: return sigmoid(summ);
	}
}

double actfnc::ActivateFunc::divActFunc(double summ)
{
	switch (currentType)
	{
	case Relu: return divRelu(summ);
	case Sigmoid: return divSigmoid(summ);
	}
}

double actfnc::ActivateFunc::revActFunc(double summ)
{
	switch (currentType)
	{
	case actfnc::Relu: return revRelu(summ);
	case actfnc::Sigmoid: return revSigmoid(summ);
	}
}

