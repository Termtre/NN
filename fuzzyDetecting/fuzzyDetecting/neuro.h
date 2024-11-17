#pragma once
#include <vector>
#include <string>
#include "activateFunc.h"

class NeuroNetwork
{
private:
	size_t numberLayers;
	size_t sizeOfInputLayer;
	size_t sizeOfOutputLayer;
	std::vector<size_t> sizeOfLayers;

	std::vector<std::vector<std::vector<double>>> weights;
	std::vector<std::vector<double>> layers;

	actfnc::ActivateFunc actFunc;
	double MSE(const std::vector<double>& answers);
	void backPropagation(double koef, const std::vector<double>& answers);

public:
	NeuroNetwork() = default;
	NeuroNetwork(size_t _numberLayers, const std::vector<size_t>& _sizeLayers);

	void forward();
	void train(double speed, size_t maxEpoch,
		const std::vector<std::vector<double>>& inputs,
		const std::vector<std::vector<double>>& answers);

	void getOutputLayer();
	void getOutputLayer(const std::string& path);
	std::vector<double> outputLayer();

	void updateInputLayer(const std::vector<double>& data);
	void updateInputLayer(const std::string& path);
	void getInputLayer();

	void updateWeights(const std::vector<std::vector<std::vector<double>>>& otherWeights);
	void updateWeights(const std::string& path);
	void generateWeights();
	void saveWeights(const std::string& path);
	void getWeights();

	void getSizeOfLayers();
	void getLayers();
};