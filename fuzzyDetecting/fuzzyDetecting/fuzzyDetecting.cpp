#include <iostream>
#include <random>
#include "neuro.h"
#include <fstream>

int maxVec(const std::vector<double>& vec);

int main()
{
    srand(time(NULL));

	std::ifstream fileInputs("mnist60kImages.txt"), fileAnswers("mnist60kLabels.txt");
	std::vector<std::vector<double>> data, answers;

	if (!fileInputs.is_open())
	{
		throw std::runtime_error("fileInputs didn't open!!!");
	}

	if (!fileAnswers.is_open())
	{
		throw std::runtime_error("fileAnswers didn't open!!!");
	}

	for (int i = 0; i < 10000; i++)
	{
		data.push_back(std::vector<double>(784));

		for (int j = 0; j < 784; j++)
		{
			fileInputs >> data.back()[j];
		}

		answers.push_back(std::vector<double>(10, 0));

		int buffer;
		fileAnswers >> buffer;
		answers.back()[buffer % 10] = 1;
	}

	//std::vector<double> test(784);
	//std::vector<double> testAns(10, 0);

	//for (int j = 0; j < 784; j++)
	//{
	//	fileInputs >> test[j];
	//}

	//int buffer;
	//fileAnswers >> buffer;
	//testAns[buffer % 10] = 1;

	//for (auto& it : test)
	//{
	//	std::cout << it << " ";
	//}

	//std::cout << std::endl;

	//for (auto& it : testAns)
	//{
	//	std::cout << it << " ";
	//}

	//std::cout << std::endl;

    NeuroNetwork NN(3, std::vector<size_t>{784, 256, 10});
	NN.updateWeights("weightsTrainMNIST10K12epoch.txt");

	int number = 11000 - 10001;
	int numberTrue = 0;

	for (int i = 10001; i < 11000; i++)
	{
		std::cout << i << std::endl;
		std::vector<double> data1(784);
		std::vector<double> answers1(10, 0);

		for (int j = 0; j < 784; j++)
		{
			fileInputs >> data1[j];
		}

		int buffer;
		fileAnswers >> buffer;
		answers1[buffer % 10] = 1;

		NN.updateInputLayer(data1);
		NN.forward();
		int pos1 = maxVec(NN.outputLayer());
		int pos2 = maxVec(answers1);

		numberTrue += (pos1 == pos2);
	}

	fileInputs.close();
	fileAnswers.close();

	std::cout << "Correct detect: " << numberTrue << " of " << number << " : " << (double)numberTrue / number << "\n";
}

int maxVec(const std::vector<double>& vec)
{
	double max = vec[0];
	int pos = 0;

	for (int i = 1; i < vec.size(); i++)
	{
		if (max < vec[i])
		{
			max = vec[i];
			pos = i;
		}
	}

	return pos;
}
