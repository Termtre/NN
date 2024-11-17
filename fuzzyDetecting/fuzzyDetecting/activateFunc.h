#pragma once

namespace actfnc
{
	enum { Relu, Sigmoid};

	class ActivateFunc
	{
	private:
		size_t currentType;

		double sigmoid(double summ);
		double divSigmoid(double summ);
		double revSigmoid(double summ);

		double relu(double summ);
		double divRelu(double summ);
		double revRelu(double summ);

	public:
		ActivateFunc();
		ActivateFunc(size_t type);
		void changeType(size_t newType);

		double actFunc(double summ);
		double divActFunc(double summ);
		double revActFunc(double summ);
	};
};

