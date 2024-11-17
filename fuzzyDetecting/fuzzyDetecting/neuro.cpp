#include "neuro.h"
#include <random>
#include <fstream>
#include <iostream>
#include <cmath>
#include <ctime>
#include <iomanip>

void NeuroNetwork::backPropagation(double koef, const std::vector<double>& answers)
{
	std::vector<double> grad;
	double alpha = 1.f;
	for (int i = numberLayers - 1; i >= 0; i--)
	{
		if (i == numberLayers - 1)
		{
			double temp = 0.;
			for (int j = 0; j < sizeOfLayers[i]; j++)
			{
				temp = answers[j] - layers[i][j];

				grad.push_back(actFunc.divActFunc(actFunc.revActFunc(layers[i][j])) * temp);
			}
		}
		else
		{
			for (int k = 0; k < grad.size(); k++)
			{
				for (int j = 0; j < sizeOfLayers[i]; j++)
				{
					weights[i][k][j] += koef * layers[i][j] * grad[k];
				}
			}

			if (i != 0)
			{
				std::vector<double> tempGrad;
				double temp = 0.;
				for (int j = 0; j < sizeOfLayers[i]; j++)
				{
					for (int k = 0; k < grad.size(); k++)
					{
						//std::cout << grad.size() << std::endl;
						temp += weights[i][k][j] * grad[k];
					}

					tempGrad.push_back(actFunc.divActFunc(actFunc.revActFunc(layers[i][j])) * temp);
				}

				grad = tempGrad;
			}
		}
	}
}

double NeuroNetwork::MSE(const std::vector<double>& answers)
{
	double temp = 0.;
	for (int i = 0; i < answers.size(); i++)
	{
		temp += (answers[i] - layers[numberLayers - 1][i]) * (answers[i] - layers[numberLayers - 1][i]);
	}
	return 0.5 * temp;
}

NeuroNetwork::NeuroNetwork(size_t _numberLayers, const std::vector<size_t>& _sizeLayers) :
	numberLayers(_numberLayers),
	sizeOfLayers(_sizeLayers)
{
	for (int i = 0; i < numberLayers - 1; i++)
	{
		sizeOfLayers[i]++;
	}

	sizeOfInputLayer = sizeOfLayers[0];
	sizeOfOutputLayer = sizeOfLayers[numberLayers - 1];

	layers.resize(numberLayers);

	for (int i = 0; i < numberLayers; i++)
	{
		if (i != numberLayers - 1)
		{
			layers[i].resize(sizeOfLayers[i] + 1);
			layers[i][sizeOfLayers[i]] = 1.;
		}
		else layers[i].resize(sizeOfLayers[i]);
	}

	weights.resize(numberLayers - 1);

	for (int i = 0; i < numberLayers - 1; i++)
	{
		weights[i].resize(sizeOfLayers[i + 1]);

		for (int j = 0; j < sizeOfLayers[i + 1]; j++)
		{
			weights[i][j].resize(sizeOfLayers[i]);
		}
	}

}

void NeuroNetwork::forward()
{
	for (int i = 1; i < numberLayers; i++)
	{
		for (int j = 0; j < sizeOfLayers[i]; j++)
		{
			double temp = 0.;

			for (int k = 0; k < sizeOfLayers[i - 1]; k++)
			{
				temp += weights[i - 1][j][k] * layers[i - 1][k];
			}

			if ((i == numberLayers - 1) || (j != sizeOfLayers[i] - 1)) layers[i][j] = actFunc.actFunc(temp);
			else layers[i][j] = 1.;
		}
	}
}

void NeuroNetwork::train(double speed, size_t maxEpoch,
	const std::vector<std::vector<double>>& inputs,
	const std::vector<std::vector<double>>& answers)
{
	generateWeights();

	//getWeights();

	std::cout << "Start learn: speed " << speed << " maxEpoch " << maxEpoch << std::endl;

	auto start = clock();

	for (int i = 0; i < maxEpoch; i++)
	{
		double temp = 0.;
		std::cout << "Epoch " << i << " " << std::endl;
		auto startEpoch = clock();
		for (int j = 0; j < inputs.size(); j++)
		{
			updateInputLayer(inputs[j]);
			forward();
			temp += MSE(answers[j]);
			backPropagation(speed, answers[j]);
			//getWeights();
		}

		auto endEpoch = clock();
		std::cout << "Time: " << (double)(endEpoch - startEpoch) / CLOCKS_PER_SEC << std::endl;
		std::cout << "Total Error: " << temp / inputs.size() << std::endl;
		std::cout << "___________________________________________________________________\n";
	}

	auto end = clock();

	//getWeights();

	std::cout << "Total Time " << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
}

void NeuroNetwork::getOutputLayer()
{
	for (const auto& it : layers[numberLayers - 1])
		std::cout << it << " ";
	std::cout << std::endl;
}

void NeuroNetwork::getOutputLayer(const std::string& path)
{
	std::ofstream potokVivoda(path);

	if (!potokVivoda.is_open()) throw std::runtime_error("void NeuroNetwork::getOutputLayer(const std::string& path): File didn't open!!!");

	for (const auto& it : layers[numberLayers - 1])
		potokVivoda << it << " ";

	potokVivoda.close();
}

std::vector<double> NeuroNetwork::outputLayer()
{
	return layers[numberLayers - 1];
}

void NeuroNetwork::updateInputLayer(const std::vector<double>& data)
{
	layers[0] = data;
	layers[0].push_back(1.f);
	sizeOfLayers[0] = layers[0].size();
	sizeOfInputLayer = layers[0].size();
}

void NeuroNetwork::updateInputLayer(const std::string& path)
{
	std::ifstream potokVvoda(path);

	if (!potokVvoda.is_open()) throw std::runtime_error("void NeuroNetwork::inputInputLayer(const std::string& path): File didn't open!!!");

	layers[0].clear();
	double buffer;

	while (potokVvoda >> buffer)
	{
		layers[0].push_back(buffer);
	}

	potokVvoda.close();

	layers[0].push_back(1.f);
	sizeOfLayers[0] = layers[0].size();
	sizeOfInputLayer = layers[0].size();
}

void NeuroNetwork::getInputLayer()
{
	for (const auto& it : layers[0])
		std::cout << it << " ";
	std::cout << std::endl;
}

void NeuroNetwork::updateWeights(const std::vector<std::vector<std::vector<double>>>& otherWeights)
{
	this->weights = otherWeights;
}

void NeuroNetwork::updateWeights(const std::string& path)
{
	std::ifstream potokVvoda(path);

	if (!potokVvoda.is_open()) throw std::runtime_error("void NeuroNetwork::updateWeights(const std::string& path): File didn't open!!!");

	for (auto& it : weights)
		for (auto& it1 : it)
			for (auto& it2 : it1)
			potokVvoda >> it2;

	potokVvoda.close();
}

void NeuroNetwork::generateWeights()
{
	for (int i = 0; i < numberLayers - 1; i++)
	{
		for (int j = 0; j < sizeOfLayers[i + 1]; j++)
		{
			for (int k = 0; k < sizeOfLayers[i]; k++)
			{
				//weights[i][j][k] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
				weights[i][j][k] = (rand() % 100) * 0.0003;
			}
		}
	}
}

void NeuroNetwork::saveWeights(const std::string& path)
{
	std::ofstream potokVivoda(path);

	if (!potokVivoda.is_open()) throw std::runtime_error("void NeuroNetwork::saveWeights(const std::string& path): File didn't open!!!");

	for (const auto& it : weights)
	{
		for (const auto& it1 : it)
		{
			for (const auto& it2 : it1)
			{
				potokVivoda << it2 << " ";
			}
			potokVivoda << "\n";
		}
		potokVivoda << "\n";
	}

	potokVivoda.close();
}

void NeuroNetwork::getWeights()
{
	for (int k = 0; k < weights.size(); k++)
	{
		for (int i = 0; i < weights[k].size(); i++)
		{
			for (int j = 0; j < weights[k][i].size(); j++)
			{
				std::cout << std::setprecision(5) << "W[" << k << "][" << i << "][" << j << "] - " << weights[k][i][j] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}

void NeuroNetwork::getSizeOfLayers()
{
	for (auto& it : sizeOfLayers)
		std::cout << it << " ";
	std::cout << std::endl;
}

void NeuroNetwork::getLayers()
{
	for (int i = 0; i < numberLayers; i++)
	{
		for (int j = 0; j < sizeOfLayers[i]; j++)
		{
			std::cout << layers[i][j] << " ";
		}
		std::cout << std::endl << std::endl;
	}
}
